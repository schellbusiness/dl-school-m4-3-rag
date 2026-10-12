#!/usr/bin/env node
/**
 * Meine deterministische RAG-Chain — Kopie von deterministic-chain.ts,
 * umgestellt auf meinen eigenen Namespace (zwei eigene PDFs).
 *
 * Feste Schritte, das Modell entscheidet nichts selbst:
 *   1. Frage einbetten       (OpenAI text-embedding-3-small, wie beim Seed)
 *   2. Retrieval             (Top-K ähnlichste Chunks aus MEINEM Namespace)
 *   3. Score-Schwelle        (nichts Passendes → Ablehnung ohne LLM-Aufruf)
 *   4. Prompt + LLM          (antwortet nur aus dem Kontext, mit Zitat)
 *
 * Usage:
 *   npm run rag:mine -- "Deine Frage"
 */

import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { loadEnv } from "../shared/load-env.js";

loadEnv(); // .env laden, BEVOR wir die Konstanten unten auslesen

const INDEX_NAME = process.env.WORKSHOP_PINECONE_INDEX ?? "handbook";
const NAMESPACE = process.env.WORKSHOP_PINECONE_NAMESPACE ?? "vanessa-docs";
const EMBEDDING_MODEL = "text-embedding-3-small"; // muss zum Seed passen!
const TOP_K = 4;
const MIN_SCORE = 0.3; // unter diesem Ähnlichkeitswert gilt ein Chunk als "passt nicht"
const REFUSAL = "Dazu steht nichts in meinen Dokumenten.";

interface Hit {
  id: string;
  score: number;
  text: string;
  doc: string;
  page?: number;
  section?: string;
  section_id?: string;
  chunk_index?: number;
}

// ---- Schritt 1+2: Retrieval aus meinem Namespace ---------------------------
async function retrieve(question: string): Promise<Hit[]> {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const embedding = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: question });

  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
  const host = (await pc.describeIndex(INDEX_NAME)).host;
  const ns = pc.index(INDEX_NAME, host).namespace(NAMESPACE);

  const result = await ns.query({
    vector: embedding.data[0].embedding,
    topK: TOP_K,
    includeMetadata: true,
  });

  return (result.matches ?? []).map((m) => {
    const meta = (m.metadata ?? {}) as Record<string, unknown>;
    return {
      id: m.id,
      score: m.score ?? 0,
      text: String(meta.chunk_text ?? ""),
      doc: String(meta.doc ?? "?"),
      page: typeof meta.page === "number" ? meta.page : undefined,
      section: typeof meta.section === "string" ? meta.section : undefined,
      section_id: typeof meta.section_id === "string" ? meta.section_id : undefined,
      chunk_index: typeof meta.chunk_index === "number" ? meta.chunk_index : undefined,
    };
  });
}

// ---- Zitatformat: [Datei · Abschnitt · S. x · Chunk n] ----------------------
function citation(h: Hit): string {
  return [
    h.doc,
    h.section,
    h.page != null ? `S. ${h.page}` : null,
    h.chunk_index != null ? `Chunk ${h.chunk_index}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function formatContext(hits: Hit[]): string {
  return hits
    .map((h, i) => `[${i + 1}] ${citation(h)} (score ${h.score.toFixed(3)})\n${h.text}`)
    .join("\n\n");
}

// ---- Schritt 3+4: Schwelle, Prompt, LLM -------------------------------------
async function main(): Promise<void> {
  if (!process.env.PINECONE_API_KEY) throw new Error("Missing PINECONE_API_KEY in .env");
  if (!process.env.OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY in .env");

  const question = process.argv.slice(2).join(" ").trim();
  if (!question) throw new Error('Bitte eine Frage mitgeben: npm run rag:mine -- "Deine Frage"');

  console.log(`Pattern: Deterministic RAG · Namespace "${NAMESPACE}"\n`);
  console.log(`Frage: ${question}\n`);

  const hits = (await retrieve(question)).filter((h) => h.score >= MIN_SCORE);

  if (!hits.length) {
    // Deterministische Ablehnung: Kein Chunk ist ähnlich genug → LLM gar nicht fragen.
    console.log("Kein Chunk über der Score-Schwelle.\n\n--- Antwort ---\n");
    console.log(REFUSAL);
    return;
  }

  console.log("Gefundener Kontext:\n");
  console.log(formatContext(hits));
  console.log();

  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const response = await llm.invoke([
    new SystemMessage(
      [
        "Du beantwortest Fragen ausschließlich anhand des Kontexts unten. Kein Vorwissen, keine Vermutungen.",
        "Antworte auf Deutsch, kurz und sachlich.",
        "Belege jede Aussage mit der Nummer des Textstücks in eckigen Klammern, z. B. [1] oder [2][3].",
        "Schreib nur die Nummer, keine Dateinamen oder Seiten.",
        `Steht die Antwort nicht im Kontext, antworte exakt: "${REFUSAL}"`,
      ].join(" "),
    ),
    new HumanMessage(`Kontext:\n${formatContext(hits)}\n\nFrage: ${question}`),
  ]);

  const raw =
    typeof response.content === "string" ? response.content : JSON.stringify(response.content);

  // Zitate deterministisch auflösen: Das Modell schreibt nur [n], der Code setzt die
  // echten Metadaten ein. So kann das Modell keine Quelle falsch abschreiben oder erfinden.
  const answer = raw.replace(/\[(\d+)\]/g, (match, n) => {
    const hit = hits[Number(n) - 1];
    return hit ? `[${citation(hit)}]` : match;
  });

  console.log("--- Antwort ---\n");
  console.log(answer);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
