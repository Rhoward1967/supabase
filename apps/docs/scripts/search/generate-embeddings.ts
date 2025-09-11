import '../utils/dotenv.js'

import { createClient, type SupabaseClient } from '@supabase/supabase-js'
import { parseArgs } from 'node:util'
import { OpenAI } from 'openai'
import { v4 as uuidv4 } from 'uuid'

import type { Section } from '../helpers.mdx.js'
import {
  type PageInfo,
  type PageSectionForEmbedding,
  type PageSectionWithEmbedding,
  type ProcessingResult,
  createBatches,
  mapEmbeddingsToSections,
  updatePageInsertionCounts,
  computePageResults,
  logFailedSectionIndexes,
} from './embeddings/utils.js'
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

function isNimbusMode(): boolean {
  return process.env.ENABLED_FEATURES_OVERRIDE_DISABLE_ALL === 'true'
}

function getPageTables() {
  const nimbus = isNimbusMode()
  return {
    pageTable: nimbus ? 'page_nimbus' : 'page',
    pageSectionTable: nimbus ? 'page_section_nimbus' : 'page_section',
  } as const
}

function requireEnvOrThrow(names: string[]): void {
  const missing = names.filter((n) => !process.env[n])
  if (missing.length) {
    throw new Error(
      `Environment variables ${missing.join(', ')} are required: skipping embeddings generation`
    )
  }
}

function initSupabase(): SupabaseClient {
  return createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SECRET_KEY!, {
    auth: { persistSession: false, autoRefreshToken: false },
  })
}

type PreparedSections = {
  allSectionsToProcess: PageSectionForEmbedding[]
  pageInfoMap: Map<number, PageInfo>
}

async function prepareSections(
  supabaseClient: SupabaseClient,
  pageTable: string,
  pageSectionTable: string,
  shouldRefresh: boolean,
  refreshVersion: string,
  refreshDate: Date
): Promise<PreparedSections> {
  const embeddingSources = await fetchAllSources()
  console.log(`Discovered ${embeddingSources.length} pages`)

  const allSectionsToProcess: PageSectionForEmbedding[] = []
  const pageInfoMap = new Map<number, PageInfo>()

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

          const { error: fetchPageError, data: existingPage } = await supabaseClient
            .from(pageTable)
            .select('id, path, checksum')
            .filter('path', 'eq', path)
            .limit(1)
            .maybeSingle()

          if (fetchPageError) throw fetchPageError

          if (!shouldRefresh && existingPage?.checksum === checksum) {
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
            if (updatePageError) throw updatePageError
            return
          }

          if (existingPage) {
            console.log(
              !shouldRefresh
                ? `[${path}] Docs have changed, removing old page sections and their embeddings`
                : `[${path}] Refresh flag set, removing old page sections and their embeddings`
            )

            const { error: deletePageSectionError } = await supabaseClient
              .from(pageSectionTable)
              .delete()
              .filter('page_id', 'eq', existingPage.id)
            if (deletePageSectionError) throw deletePageSectionError
          }

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
          if (upsertPageError) throw upsertPageError

          console.log(`[${path}] Preparing ${sections.length} page sections for processing`)

          pageInfoMap.set(page.id, {
            pageId: page.id,
            path,
            checksum,
            sectionsCount: sections.length,
          })

          const sectionsForBatching = sections.map(({ slug, heading, content }) => ({
            pageId: page.id,
            path,
            slug,
            heading,
            content,
            input: content.replace(/\n/g, ' '),
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

  return { allSectionsToProcess, pageInfoMap }
}

async function processAndInsertEmbeddings(
  openai: OpenAI,
  supabaseClient: SupabaseClient,
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
    try {
      const batchResult = await processEmbeddingBatch(
        openai,
        batch,
        batchIndex,
        embeddingBatches.length
      )

      result.totalSectionsProcessed += batchResult.processedCount

      if (batchResult.sectionsWithEmbeddings.length > 0) {
        const insertedCount = await insertSectionBatch(
          supabaseClient,
          pageSectionTable,
          batchResult.sectionsWithEmbeddings
        )
        result.totalSectionsInserted += insertedCount
        updatePageInsertionCounts(pageSectionsInserted, batchResult.sectionsWithEmbeddings)
      }

      // Mark failed section pages
      batchResult.failedSectionIndexes.forEach((i) => {
        result.failedPages.add(batch[i].pageId)
      })
    } catch (error) {
      console.error(`Batch ${batchIndex + 1} completely failed:`, error)
      batch.forEach((section) => result.failedPages.add(section.pageId))
    }

    if (batchIndex < embeddingBatches.length - 1) {
      await delay(CONFIG.OPENAI_BASE_DELAY_MS)
    }
  }

  computePageResults(pageInfoMap, pageSectionsInserted, result)

  return result
}

type BatchEmbeddingResult = {
  sectionsWithEmbeddings: PageSectionWithEmbedding[]
  failedSectionIndexes: number[]
  processedCount: number
}

async function processEmbeddingBatch(
  openai: OpenAI,
  batch: PageSectionForEmbedding[],
  batchIndex: number,
  totalBatches: number
): Promise<BatchEmbeddingResult> {
  const inputs = batch.map((section) => section.input)
  console.log(
    `Processing embedding batch ${batchIndex + 1}/${totalBatches} (${inputs.length} sections)`
  )

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

  const { sectionsWithEmbeddings, failedSectionIndexes } = mapEmbeddingsToSections(
    batch,
    inputs,
    embeddingResponse.data
  )
  logFailedSectionIndexes(batch, inputs, failedSectionIndexes)

  return {
    sectionsWithEmbeddings,
    failedSectionIndexes,
    processedCount: inputs.length,
  }
}

async function insertSectionBatch(
  supabaseClient: SupabaseClient,
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

  if (isNimbusMode()) {
    console.log('Running in Nimbus mode - will filter content based on disabled feature flags')
  }

  requireEnvOrThrow([
    'DOCS_GITHUB_APP_ID',
    'DOCS_GITHUB_APP_INSTALLATION_ID',
    'DOCS_GITHUB_APP_PRIVATE_KEY',
    'NEXT_PUBLIC_MISC_ANON_KEY',
    'NEXT_PUBLIC_MISC_URL',
    'NEXT_PUBLIC_SUPABASE_URL',
    'OPENAI_API_KEY',
    'SUPABASE_SECRET_KEY',
  ])

  const supabaseClient = initSupabase()

  const refreshVersion = uuidv4()
  const refreshDate = new Date()

  const { pageTable, pageSectionTable } = getPageTables()
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

  console.log(
    shouldRefresh
      ? 'Refresh flag set, re-generating all pages'
      : 'Checking which pages are new or have changed'
  )

  const { allSectionsToProcess, pageInfoMap } = await prepareSections(
    supabaseClient,
    pageTable,
    pageSectionTable,
    shouldRefresh,
    refreshVersion,
    refreshDate
  )

  console.log(`Embedding processing and insertion for ${allSectionsToProcess.length} sections`)

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

  console.log(`\nUpdating checksums for ${processingResult.successfulPages.size} successful pages`)
  const successfulChecksumUpdates = await updateSuccessfulChecksums(
    supabaseClient,
    pageTable,
    pageInfoMap,
    processingResult
  )
  console.log(
    `Successfully updated checksums for ${successfulChecksumUpdates}/${processingResult.successfulPages.size} successful pages`
  )

  logFailedPages(pageInfoMap, processingResult)

  await purgeOldPages(supabaseClient, pageTable, refreshVersion)

  console.log('Embedding generation complete')
}

async function updateSuccessfulChecksums(
  supabaseClient: SupabaseClient,
  pageTable: string,
  pageInfoMap: Map<number, PageInfo>,
  processingResult: ProcessingResult
): Promise<number> {
  let successfulChecksumUpdates = 0
  const pageIds = Array.from(processingResult.successfulPages)
  const batches = createBatches(pageIds, CONFIG.SOURCE_CONCURRENCY)

  for (const batch of batches) {
    const results = await Promise.all(
      batch.map(async (pageId) => {
        const pageInfo = pageInfoMap.get(pageId)
        if (!pageInfo) {
          console.error(`Missing page info for pageId ${pageId}`)
          return 0
        }

        try {
          const { error: updatePageError } = await supabaseClient
            .from(pageTable)
            .update({ checksum: pageInfo.checksum })
            .eq('id', pageId)
          if (updatePageError) {
            console.error(`Failed to update checksum for page ${pageInfo.path}:`, updatePageError)
            return 0
          }
          return 1
        } catch (error) {
          console.error(`Error updating checksum for page ${pageInfo.path}:`, error)
          return 0
        }
      })
    )

    successfulChecksumUpdates += results.reduce((sum, x) => sum + x, 0)
  }

  return successfulChecksumUpdates
}

function logFailedPages(pageInfoMap: Map<number, PageInfo>, processingResult: ProcessingResult) {
  if (processingResult.failedPages.size === 0) return
  console.log(`\nFailed pages:`)
  for (const pageId of processingResult.failedPages) {
    const pageInfo = pageInfoMap.get(pageId)
    if (pageInfo) console.log(`  - ${pageInfo.path}`)
  }
}

async function purgeOldPages(
  supabaseClient: SupabaseClient,
  pageTable: string,
  refreshVersion: string
) {
  console.log(`Removing old pages and their sections`)
  const { error: deletePageError } = await supabaseClient
    .from(pageTable)
    .delete()
    .filter('version', 'neq', refreshVersion)
  if (deletePageError) throw deletePageError
}

async function main() {
  await generateEmbeddings()
}

main().catch((err) => {
  console.error(err)

  // Exit with non-zero code
  process.exit(1)
})
