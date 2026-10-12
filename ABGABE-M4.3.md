# Abgabe M4.3 – LLM-Integration & Datenanbindung

## M4.3.1: Dein eigener Index

Meine zwei PDFs: Pizza.pdf, Rotkaeppchen.pdf
(Wikipedia-Artikel „Pizza" und „Rotkäppchen", als PDF exportiert; 11 + 18 Seiten)

Ein Chunk mit Metadaten:

```json
{
  "id": "rotkaeppchen-c003",
  "doc": "Rotkaeppchen.pdf",
  "chunk_index": 3,
  "page": 2,
  "section": "Inhalt",
  "section_id": "3",
  "chunk_text": "Inhalt — Ein kleines Mädchen, dem seine Großmutter einst eine rote Kappe geschenkt hatte, daher Rotkäppchen genannt, wird von der Mutter zu der in einem Haus im Wald wohnenden, bettlägerig kranken Großmutter mit einem Korb mit Leckereien (Kuchen und Wein) geschickt. Die Mutter warnt Rotkäppchen eindringlich, es solle nicht vom Weg abgehen. Im Wald lässt es sich auf ein Gespräch mit einem Wolf ein. Dieser horcht Rotkäppchen aus und macht es auf die schönen Blumen auf einer nahen Wiese aufmerksam, worauf Rotkäppchen beschließt, noch einen Blumenstrauß zu pflücken, der Warnung der Mutter zum Trotz. Der Wolf eilt geradewegs zur Großmutter und verschlingt sie. Er legt sich in deren Nachthemd in ihr Bett und wartet auf Rotkäppchen. Bald darauf erreicht Rotkäppchen das Haus, tritt ein, und begibt sich in (bei Perrault) bzw. an (bei den Brüdern Grimm) Großmutters Bett. Dort wundert sich Rotkäppchen über die Gestalt seiner Großmutter, erkennt aber den Wolf nicht, bevor es ebenfalls verschlungen wird. Bei Perrault endet das Märchen hier. Großmutter und Rotkäppchen werden bei den Brüdern Grimm vom Jäger aus dem Bauch des schlafenden Wolfes befreit."
}
```

Metadaten: section = Überschrift, doc = Quelldatei, chunk_index = Chunk-Nummer (zusätzlich page, section_id).

Mein Namespace: vanessa-docs (Index „handbook") · 63 Vektoren
Seed-Ausgabe: „✓ Synced 63 vectors to Pinecone" (Screenshot anhängen)

Chunking-Skript: scripts/chunk-pdfs.mjs (pdf.js; Überschriften über Schriftgröße erkannt,
Kopf-/Fußzeilen, Bildunterschriften und Quellenabschnitte entfernt, Chunks ≤ 1.200 Zeichen an Satzgrenzen).

## M4.3.2: Fragen an deine Dokumente

Sechs Fragen an meine Chain:

Frage: Wann und von wem soll die erste Pizza Margherita gebacken worden sein?
Antwort der Chain (gekürzt): Am 11. Juni 1889 vom Pizzaiolo Raffaele Esposito in Neapel.
Zitat bzw. Ablehnung: [Pizza.pdf · 19. Jahrhundert / Pizza Margherita · S. 2 · Chunk 5]

Frage: Bei welcher Temperatur wird eine Pizza gebacken?
Antwort der Chain (gekürzt): Bei möglichst hoher Temperatur von 400 bis 500 °C.
Zitat bzw. Ablehnung: [Pizza.pdf · Zubereitung · S. 3 · Chunk 7]

Frage: Wer rettet Rotkäppchen und die Großmutter bei den Brüdern Grimm?
Antwort der Chain (gekürzt): Ein Jäger, aus dem Bauch des schlafenden Wolfes.
Zitat bzw. Ablehnung: [Rotkaeppchen.pdf · Inhalt · S. 2 · Chunk 3]

Frage: Was passiert am Ende mit dem Wolf?
Antwort der Chain (gekürzt): Er wird mit Steinen gefüllt, kann nicht fliehen und stirbt.
Zitat bzw. Ablehnung: [Rotkaeppchen.pdf · Inhalt · S. 2 · Chunk 4]

Frage: Wie endet das Märchen in der Version von Charles Perrault?
Antwort der Chain (gekürzt): Großmutter und Rotkäppchen werden gefressen und nicht gerettet.
Zitat bzw. Ablehnung: [Rotkaeppchen.pdf · Charles Perraults petit chaperon rouge · S. 6 · Chunk 14]

Frage: Wer hat die Fußball-WM 2014 gewonnen?
Antwort der Chain (gekürzt): –
Zitat bzw. Ablehnung: Ablehnung: „Dazu steht nichts in meinen Dokumenten." (kein Chunk über Score-Schwelle 0,3, LLM wurde nicht aufgerufen)

## Meine angepasste Chain: example/rag/my-deterministic-chain.ts

Änderungen gegenüber deterministic-chain.ts:
- Retrieval direkt auf meinen Namespace vanessa-docs (eigene Pinecone-Abfrage statt shared Store)
- Score-Schwelle 0,3: ohne passenden Chunk deterministische Ablehnung ohne LLM-Aufruf
- Prompt auf Deutsch, nur aus dem Kontext antworten, feste Ablehnungsformel
- Zitate: LLM nennt nur [n], der Code setzt die echten Metadaten ein → [Datei · Abschnitt · S. x · Chunk n]

```ts
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
```

## M4.3.3: Golden Set (optional)

Golden Set mit zehn Frage-Antwort-Paaren (maschinenlesbar in data/golden-set.json,
prüfbar mit `npm run eval:golden`):

| # | Frage | Richtige Antwort | Fundstelle |
|---|---|---|---|
| 1 | Seit wann ist die Pizza mit Tomaten nachgewiesen? | Seit etwa der Mitte des 18. Jahrhunderts, als die Tomate in Süditalien populär wurde. | Pizza.pdf · Geschichte 18. Jahrhundert · S. 2 · Chunk 4 |
| 2 | Welche Zutaten hat die Pizza Margherita und wofür stehen ihre Farben? | Basilikum (grün), Mozzarella (weiß) und Tomaten (rot), die italienischen Nationalfarben. | Pizza.pdf · 19. Jahrhundert / Pizza Margherita · S. 2 · Chunk 5 |
| 3 | Wann und wo wurde die erste Pizzeria in Deutschland eröffnet? | Im März 1952 in Würzburg (Sabbie di Capri, Nicolino di Camillo). | Pizza.pdf · 20. Jahrhundert · S. 3 · Chunk 6 |
| 4 | Seit wann gehört die neapolitanische Pizzabäckerkunst zum UNESCO-Kulturerbe? | Seit 2017 (immaterielles Kulturerbe der Menschheit). | Pizza.pdf · Pizza · S. 1 · Chunk 1 |
| 5 | Womit ist eine Pizza Hawaii belegt? | Mit Kochschinken und Ananas; sie ist nordamerikanischen Ursprungs. | Pizza.pdf · US-amerikanische Pizza · S. 6 · Chunk 13 |
| 6 | Wie hieß die eigene Pizza-Variante der DDR-Gastronomie? | Krusta – auf dem Blech gebacken, in handgroße Rechtecke geschnitten, 1976 in Leipzig vorgestellt. | Pizza.pdf · Pizza in der DDR · S. 7 · Chunk 16 |
| 7 | Unter welcher Nummer steht Rotkäppchen in den Kinder- und Hausmärchen der Brüder Grimm? | Unter Nummer 26 (KHM 26). | Rotkaeppchen.pdf · Rotkäppchen · S. 1 · Chunk 1 |
| 8 | Was bringt Rotkäppchen der Großmutter im Korb mit? | Kuchen und Wein. | Rotkaeppchen.pdf · Inhalt · S. 2 · Chunk 3 |
| 9 | Aus welchem anderen Grimm-Märchen ist das Ende mit den Steinen im Bauch des Wolfes entliehen? | Aus „Der Wolf und die sieben jungen Geißlein“. | Rotkaeppchen.pdf · Erste deutsche Fassungen · S. 6 · Chunk 16 |
| 10 | Wie viele Kalorien hat eine Pizza Margherita? | Ablehnung: Dazu steht nichts in meinen Dokumenten. | – (nicht in den Dokumenten) |
