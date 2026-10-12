#!/usr/bin/env node
/**
 * Prüft die Chain gegen das Golden Set (data/golden-set.json).
 *
 * Usage:  npm run eval:golden
 *
 * Pro Frage wird die echte Chain aufgerufen und zweimal geprüft:
 *   1. Inhalt:  Stehen alle Stichworte aus must_contain in der Antwort?
 *   2. Quelle:  Zitiert die Antwort den erwarteten Chunk (Datei + Chunk-Nummer)?
 *              Bei expected_chunk_id = null muss die Chain ablehnen.
 */
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const golden = JSON.parse(readFileSync("data/golden-set.json", "utf8"));
const corpus = JSON.parse(readFileSync("data/my-corpus.json", "utf8"));
const byId = Object.fromEntries(corpus.chunks.map((c) => [c.id, c]));

let passed = 0;
golden.questions.forEach((q, i) => {
  const output = execFileSync("npx", ["tsx", "example/rag/my-deterministic-chain.ts", q.question], {
    env: { ...process.env, WORKSHOP_PINECONE_NAMESPACE: golden.namespace },
    encoding: "utf8",
  });
  const answer = output.split("--- Antwort ---").pop().trim();

  const contentOk = q.must_contain.every((w) => answer.toLowerCase().includes(w.toLowerCase()));
  let sourceOk;
  if (q.expected_chunk_id === null) {
    sourceOk = !/\[[^\]]*Chunk \d+\]/.test(answer); // Ablehnung: keine Quelle zitiert
  } else {
    const c = byId[q.expected_chunk_id];
    sourceOk = answer.includes(`${c.doc} ·`) && answer.includes(`Chunk ${c.chunk_index}]`);
  }

  const ok = contentOk && sourceOk;
  if (ok) passed++;
  console.log(`${ok ? "✅" : "❌"} ${i + 1}. ${q.question}`);
  if (!ok) {
    console.log(`   Inhalt ${contentOk ? "ok" : "FEHLT: " + q.must_contain.join(", ")} · Quelle ${sourceOk ? "ok" : "erwartet " + q.source}`);
    console.log(`   Antwort: ${answer.slice(0, 300)}`);
  }
});

console.log(`\n${passed}/${golden.questions.length} bestanden`);
process.exit(passed === golden.questions.length ? 0 : 1);
