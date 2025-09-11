import '../utils/dotenv.js'

import { createClient, type SupabaseClient } from '@supabase/supabase-js'
import { parseArgs } from 'node:util'
import { OpenAI } from 'openai'
import { v4 as uuidv4 } from 'uuid'

import { type DatabaseCorrected } from '../../lib/supabase.js'
import type { Section } from '../helpers.mdx.js'
import { fetchAllSources } from './sources/index.js'

const CONFIG = {
  // OpenAI settings
  EMBEDDING_MODEL: 'text-embedding-ada-002' as const,
  EMBEDDING_DIMENSION: 1536, // Keep in sync with EMBEDDING_MODEL
  OPENAI_BATCH_SIZE: 128,
  OPENAI_MAX_RETRIES: 3,
  OPENAI_BASE_DELAY_MS: 500,
  OPENAI_MAX_CONCURRENCY: 3,

  // Supabase settings
  SUPABASE_BATCH_SIZE: 500,
  SUPABASE_MAX_RETRIES: 2,
  SUPABASE_BASE_DELAY_MS: 100,

  // Processing settings
  SOURCE_CONCURRENCY: 10,
} as const

/**
 * Create batches of batchSize from an array
 */
function createBatches<T>(array: T[], batchSize: number): T[][] {
  const batches: T[][] = []
  for (let i = 0; i < array.length; i += batchSize) {
    batches.push(array.slice(i, i + batchSize))
  }
  return batches
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function exponentialBackoff(attempt: number, baseDelay: number, maxDelay: number = 30000): number {
  const exponentialDelay = baseDelay * Math.pow(2, attempt)
  const jitter = (Math.random() - 0.5) * 0.1 * exponentialDelay
  return Math.min(Math.max(0, exponentialDelay + jitter), maxDelay)
}

async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number,
  baseDelay: number,
  operationName: string
): Promise<T> {
  let lastError: Error

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation()
    } catch (error) {
      lastError = error as Error

      if (attempt === maxRetries) {
        console.error(`${operationName} failed after ${maxRetries + 1} attempts:`, lastError)
        throw lastError
      }

      const delayMs = exponentialBackoff(attempt, baseDelay)
      console.warn(
        `${operationName} attempt ${attempt + 1} failed, retrying in ${delayMs}ms:`,
        lastError.message
      )
      await delay(delayMs)
    }
  }

  throw lastError!
}

interface PageInfo {
  pageId: number
  path: string
  checksum: string
  sectionsCount: number
}

interface PageSectionForEmbedding {
  pageId: number
  path: string
  slug: string
  heading: string
  content: string
  /**
   * Processed content for embedding
   */
  input: string
  ragIgnore: boolean
}

interface PageSectionWithEmbedding extends PageSectionForEmbedding {
  embedding: number[]
}

interface ProcessingResult {
  successfulPages: Set<number>
  failedPages: Set<number>
  totalSectionsProcessed: number
  totalSectionsInserted: number
}

async function processAndInsertEmbeddings(
  openai: OpenAI,
  supabaseClient: SupabaseClient<DatabaseCorrected>,
  pageSectionTable: string,
  allSections: PageSectionForEmbedding[],
  pageInfoMap: Map<number, PageInfo>
): Promise<ProcessingResult> {
  if (allSections.length === 0) {
    return {
      successfulPages: new Set(),
      failedPages: new Set(),
      totalSectionsProcessed: 0,
      totalSectionsInserted: 0,
    }
  }

  console.log(`Processing ${allSections.length} sections with embeddings + insertion`)

  const embeddingBatches = createBatches(allSections, CONFIG.OPENAI_BATCH_SIZE)
  const result: ProcessingResult = {
    successfulPages: new Set(),
    failedPages: new Set(),
    totalSectionsProcessed: 0,
    totalSectionsInserted: 0,
  }

  // Track sections inserted per page
  const pageSectionsInserted = new Map<number, number>()

  for (let batchIndex = 0; batchIndex < embeddingBatches.length; batchIndex++) {
    const batch = embeddingBatches[batchIndex]
    const inputs = batch.map((section) => section.input)

    console.log(
      `Processing embedding batch ${batchIndex + 1}/${embeddingBatches.length} (${inputs.length} sections)`
    )

    try {
      const embeddingResponse = await withRetry(
        () =>
          openai.embeddings.create({
            model: CONFIG.EMBEDDING_MODEL,
            input: inputs,
          }),
        CONFIG.OPENAI_MAX_RETRIES,
        CONFIG.OPENAI_BASE_DELAY_MS,
        `OpenAI embedding batch ${batchIndex + 1}`
      )

      if (embeddingResponse.data.length !== inputs.length) {
        console.error(
          `Warning: Expected ${inputs.length} embeddings but got ${embeddingResponse.data.length} for batch ${batchIndex + 1}`
        )
      }

      const sectionsWithEmbeddings: PageSectionWithEmbedding[] = []
      const failedSectionIndexes: number[] = []

      for (let i = 0; i < inputs.length; i++) {
        if (i < embeddingResponse.data.length && embeddingResponse.data[i]?.embedding) {
          const embeddingData = embeddingResponse.data[i]
          sectionsWithEmbeddings.push({
            ...batch[i],
            embedding: embeddingData.embedding,
          })
        } else {
          failedSectionIndexes.push(i)
          result.failedPages.add(batch[i].pageId)
        }
      }

      result.totalSectionsProcessed += inputs.length

      if (sectionsWithEmbeddings.length > 0) {
        const insertedCount = await insertSectionBatch(
          supabaseClient,
          pageSectionTable,
          sectionsWithEmbeddings
        )
        result.totalSectionsInserted += insertedCount

        // Track insertions per page
        sectionsWithEmbeddings.forEach((section) => {
          const current = pageSectionsInserted.get(section.pageId) || 0
          pageSectionsInserted.set(section.pageId, current + 1)
        })
      }

      // Log failed sections
      failedSectionIndexes.forEach((i) => {
        console.error(
          `Failed section: ${batch[i].path}#${batch[i].slug} (content: "${inputs[i]?.slice(0, 50)}...")`
        )
      })
    } catch (error) {
      console.error(`Batch ${batchIndex + 1} completely failed:`, error)

      // Mark all pages in this batch as failed
      batch.forEach((section) => {
        result.failedPages.add(section.pageId)
      })
    }

    // Add delay between batches
    if (batchIndex < embeddingBatches.length - 1) {
      await delay(CONFIG.OPENAI_BASE_DELAY_MS)
    }
  }

  // Determine successful pages (all expected sections inserted)
  for (const [pageId, pageInfo] of pageInfoMap) {
    const insertedCount = pageSectionsInserted.get(pageId) || 0
    if (insertedCount === pageInfo.sectionsCount && !result.failedPages.has(pageId)) {
      result.successfulPages.add(pageId)
    } else if (insertedCount > 0) {
      // Partial success is still a failure
      result.failedPages.add(pageId)
      console.warn(
        `Page ${pageInfo.path}: inserted ${insertedCount}/${pageInfo.sectionsCount} sections`
      )
    }
  }

  return result
}

async function insertSectionBatch(
  supabaseClient: any,
  pageSectionTable: string,
  sectionsWithEmbeddings: PageSectionWithEmbedding[]
): Promise<number> {
  if (sectionsWithEmbeddings.length === 0) {
    return 0
  }

  const pageSectionsToInsert = sectionsWithEmbeddings.map((section) => ({
    page_id: section.pageId,
    slug: section.slug,
    heading: section.heading,
    content: section.content,
    embedding: section.embedding,
    rag_ignore: section.ragIgnore,
  }))

  await withRetry(
    async () => {
      const { error } = await supabaseClient.from(pageSectionTable).insert(pageSectionsToInsert)

      if (error) {
        throw new Error(`Supabase insert error: ${error.message}`)
      }
    },
    CONFIG.SUPABASE_MAX_RETRIES,
    CONFIG.SUPABASE_BASE_DELAY_MS,
    `Insert batch of ${sectionsWithEmbeddings.length} sections`
  )

  return sectionsWithEmbeddings.length
}

const args = parseArgs({
  options: {
    refresh: {
      type: 'boolean',
    },
  },
})

async function generateEmbeddings() {
  const shouldRefresh = Boolean(args.values.refresh)
  const isNimbusMode = process.env.ENABLED_FEATURES_OVERRIDE_DISABLE_ALL === 'true'

  const pageTable = isNimbusMode ? 'page_nimbus' : 'page'
  const pageSectionTable = isNimbusMode ? 'page_section_nimbus' : 'page_section'

  if (isNimbusMode) {
    console.log('Running in Nimbus mode - will filter content based on disabled feature flags')
  }

  const requiredEnvVars = [
    'DOCS_GITHUB_APP_ID',
    'DOCS_GITHUB_APP_INSTALLATION_ID',
    'DOCS_GITHUB_APP_PRIVATE_KEY',
    'NEXT_PUBLIC_MISC_ANON_KEY',
    'NEXT_PUBLIC_MISC_URL',
    'NEXT_PUBLIC_SUPABASE_URL',
    'OPENAI_API_KEY',
    'SUPABASE_SECRET_KEY',
  ]

  const missingEnvVars = requiredEnvVars.filter((name) => !process.env[name])
  if (missingEnvVars.length > 0) {
    throw new Error(
      `Environment variables ${missingEnvVars.join(
        ', '
      )} are required: skipping embeddings generation`
    )
  }

  const supabaseClient = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SECRET_KEY!,
    {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    }
  )

  // Use this version to track which pages to purge
  // after the refresh
  const refreshVersion = uuidv4()
  const refreshDate = new Date()

  const embeddingSources = await fetchAllSources()
  console.log(`Discovered ${embeddingSources.length} pages`)

  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

  if (!shouldRefresh) {
    console.log('Checking which pages are new or have changed')
  } else {
    console.log('Refresh flag set, re-generating all pages')
  }

  // Phase 1: Collect all pages that need processing and prepare sections
  const allSectionsToProcess: PageSectionForEmbedding[] = []
  const pageInfoMap = new Map<number, PageInfo>()

  // Limit concurrency when processing embedding sources to reduce DB/API pressure
  for (const sourceBatch of createBatches(embeddingSources, CONFIG.SOURCE_CONCURRENCY)) {
    await Promise.all(
      sourceBatch.map(async (embeddingSource) => {
        const { type, source, path } = embeddingSource

        try {
        const {
          checksum,
          sections,
          meta = {},
          ragIgnore = false,
        }: {
          checksum: string
          sections: Section[]
          ragIgnore?: boolean
          meta?: Record<string, unknown>
        } = await embeddingSource.process()

        // Check for existing page in DB and compare checksums
        const { error: fetchPageError, data: existingPage } = await supabaseClient
          .from(pageTable)
          .select('id, path, checksum')
          .filter('path', 'eq', path)
          .limit(1)
          .maybeSingle()

        if (fetchPageError) {
          throw fetchPageError
        }

        // We use checksum to determine if this page & its sections need to be regenerated
        if (!shouldRefresh && existingPage?.checksum === checksum) {
          // No content/embedding update required on this page
          // Update other meta info
          const { error: updatePageError } = await supabaseClient
            .from(pageTable)
            .update({
              type,
              source,
              meta,
              version: refreshVersion,
              last_refresh: refreshDate,
            })
            .filter('id', 'eq', existingPage.id)

          if (updatePageError) {
            throw updatePageError
          }

          return
        }

        if (existingPage) {
          if (!shouldRefresh) {
            console.log(
              `[${path}] Docs have changed, removing old page sections and their embeddings`
            )
          } else {
            console.log(
              `[${path}] Refresh flag set, removing old page sections and their embeddings`
            )
          }

          const { error: deletePageSectionError } = await supabaseClient
            .from(pageSectionTable)
            .delete()
            .filter('page_id', 'eq', existingPage.id)

          if (deletePageSectionError) {
            throw deletePageSectionError
          }
        }

        // Create/update page record. Intentionally clear checksum until we
        // have successfully generated all page sections.
        const { error: upsertPageError, data: page } = await supabaseClient
          .from(pageTable)
          .upsert(
            {
              checksum: null,
              path,
              type,
              source,
              meta,
              content: embeddingSource.extractIndexedContent(),
              version: refreshVersion,
              last_refresh: refreshDate,
            },
            { onConflict: 'path' }
          )
          .select()
          .limit(1)
          .single()

        if (upsertPageError) {
          throw upsertPageError
        }

        console.log(`[${path}] Preparing ${sections.length} page sections for processing`)

        pageInfoMap.set(page.id, {
          pageId: page.id,
          path,
          checksum,
          sectionsCount: sections.length,
        })

        // Collect sections for global processing
        const sectionsForBatching = sections.map(({ slug, heading, content }) => ({
          pageId: page.id,
          path,
          slug,
          heading,
          content,
          input: content.replace(/\n/g, ' '), // OpenAI recommends replacing newlines with spaces
          ragIgnore,
        }))

        allSectionsToProcess.push(...sectionsForBatching)
      } catch (err) {
        console.error(
          `Page '${path}' or one/multiple of its page sections failed to store properly. Page has been marked with null checksum to indicate that it needs to be re-generated.`
        )
        console.error(err)
      }
      })
    )
  }

  // Phase 2: Process embeddings and insert with streaming (no memory accumulation)
  console.log(
    `\nPhase 2: Streaming embedding processing and insertion for ${allSectionsToProcess.length} sections`
  )
  let processingResult: ProcessingResult
  try {
    processingResult = await processAndInsertEmbeddings(
      openai,
      supabaseClient,
      pageSectionTable,
      allSectionsToProcess,
      pageInfoMap
    )
    console.log(
      `Processing complete: ${processingResult.totalSectionsInserted}/${processingResult.totalSectionsProcessed} sections inserted successfully`
    )
    console.log(
      `Page summary: ${processingResult.successfulPages.size} successful, ${processingResult.failedPages.size} failed`
    )
  } catch (error) {
    console.error('Critical error during embedding processing:', error)
    console.log('Exiting due to complete processing failure')
    return
  }

  // Phase 3: Update checksums ONLY for successful pages
  console.log(
    `\nPhase 3: Updating checksums for ${processingResult.successfulPages.size} successful pages`
  )
  let successfulChecksumUpdates = 0

  for (const pageId of processingResult.successfulPages) {
    const pageInfo = pageInfoMap.get(pageId)
    if (!pageInfo) {
      console.error(`Missing page info for pageId ${pageId}`)
      continue
    }

    try {
      const { error: updatePageError } = await supabaseClient
        .from(pageTable)
        .update({ checksum: pageInfo.checksum })
        .eq('id', pageId)

      if (updatePageError) {
        console.error(`Failed to update checksum for page ${pageInfo.path}:`, updatePageError)
      } else {
        successfulChecksumUpdates++
      }
    } catch (error) {
      console.error(`Error updating checksum for page ${pageInfo.path}:`, error)
    }
  }

  console.log(
    `Successfully updated checksums for ${successfulChecksumUpdates}/${processingResult.successfulPages.size} successful pages`
  )

  // Log failed pages (will be retried on next run due to null checksum)
  if (processingResult.failedPages.size > 0) {
    console.log(`\nFailed pages (will be retried next run):`)
    for (const pageId of processingResult.failedPages) {
      const pageInfo = pageInfoMap.get(pageId)
      if (pageInfo) {
        console.log(`  - ${pageInfo.path}`)
      }
    }
  }

  console.log(`Removing old pages and their sections`)

  // Delete pages that have been removed (and their sections via cascade)
  const { error: deletePageError } = await supabaseClient
    .from(pageTable)
    .delete()
    .filter('version', 'neq', refreshVersion)

  if (deletePageError) {
    throw deletePageError
  }

  console.log('Embedding generation complete')
}

async function main() {
  await generateEmbeddings()
}

main().catch((err) => {
  console.error(err)

  // Exit with non-zero code
  process.exit(1)
})
