# Exercise M4.3: LLM-Integration & Datenanbindung

Eigene RAG-Pipeline mit zwei eigenen PDFs, aufgebaut auf dem DL-School-Agentic-Workshop-Repo.

**Die vollständige Abgabe mit Chunk-Beispiel, Namespace, 6 Fragen und Golden Set steht in [`ABGABE-M4.3.md`](ABGABE-M4.3.md).**

## Wichtige Dateien

| Aufgabe | Datei | Was sie macht |
|---|---|---|
| M4.3.1 | [`scripts/chunk-pdfs.mjs`](scripts/chunk-pdfs.mjs) | PDFs → Chunks mit Metadaten (Überschrift, Quelldatei, Chunk-Nummer, Seite) |
| M4.3.1 | [`data/my-pdfs/`](data/my-pdfs) | Die zwei PDFs: `Pizza.pdf`, `Rotkaeppchen.pdf` (Wikipedia) |
| M4.3.1 | [`data/my-corpus.json`](data/my-corpus.json) | 63 Chunks im Format von `handbook-corpus.json` |
| M4.3.1 | [`seed/seed-handbook-vectors.mjs`](seed/seed-handbook-vectors.mjs) | Seed: eigener Corpus über `WORKSHOP_CORPUS`, Fix für Pinecone SDK v9 (`upsert({ records })`) |
| M4.3.2 | [`example/rag/my-deterministic-chain.ts`](example/rag/my-deterministic-chain.ts) | **Meine angepasste Chain** auf Namespace `vanessa-docs` |
| M4.3.3 | [`data/golden-set.json`](data/golden-set.json) | Golden Set: 10 Frage-Antwort-Paare mit Fundstelle |
| M4.3.3 | [`scripts/eval-golden.mjs`](scripts/eval-golden.mjs) | Prüft die Chain automatisch gegen das Golden Set (10/10 bestanden) |

## Ausführen

```bash
npm install
cp env.example .env            # PINECONE_API_KEY und OPENAI_API_KEY eintragen

npm run chunk -- data/my-pdfs/Pizza.pdf data/my-pdfs/Rotkaeppchen.pdf
npm run seed:mine              # → Namespace "vanessa-docs", 63 Vektoren
npm run rag:mine -- "Wer rettet Rotkäppchen und die Großmutter bei den Brüdern Grimm?"
npm run eval:golden            # → 10/10 bestanden
```

## Wie die Chain arbeitet

1. Die Frage wird eingebettet (`text-embedding-3-small`).
2. Die Top-4 Chunks werden aus dem Namespace `vanessa-docs` abgerufen.
3. Liegt kein Chunk über dem Score 0,3, folgt eine Ablehnung ohne LLM-Aufruf.
4. Sonst antwortet `gpt-4o-mini` nur aus dem Kontext und nennt die Quelle als `[n]`.
5. Der Code ersetzt `[n]` durch die echten Metadaten: `[Datei · Abschnitt · S. x · Chunk n]`.

Das ursprüngliche Workshop-README steht in [`WORKSHOP-README.md`](WORKSHOP-README.md).
