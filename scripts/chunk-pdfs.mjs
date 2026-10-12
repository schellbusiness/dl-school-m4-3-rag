#!/usr/bin/env node
/**
 * PDFs → Chunks mit Metadaten, im Format von data/handbook-corpus.json.
 *
 * Usage:
 *   npm run chunk -- data/my-pdfs/datei1.pdf data/my-pdfs/datei2.pdf
 *   npm run chunk -- a.pdf b.pdf --out data/my-corpus.json
 *
 * Ablauf (jeder Schritt ist unten eine eigene Funktion):
 *   1. readLines()       PDF → Textzeilen, jeweils mit Seite und Schriftgröße
 *   2. dropRepeating()   Kopf-/Fußzeilen und Seitenzahlen entfernen
 *   3. toSections()      Überschriften erkennen (größere Schrift) → Abschnitte
 *   4. toChunks()        Abschnitte in ~MAX_CHARS große Stücke an Satzgrenzen teilen
 *   5. main()            alles als JSON speichern
 */

import { readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

// ---- Stellschrauben -------------------------------------------------------
const MAX_CHARS = 1200; // Zielgröße eines Chunks (ca. 200 Wörter)
const MIN_CHARS = 200; // kleinere Reste werden an den vorherigen Chunk gehängt
const HEADING_FACTOR = 1.15; // Überschrift = Schrift mind. 15 % größer als Fließtext
const MAX_HEADING_LEN = 120; // längere "große" Zeilen sind eher Zitate/Banner
const SMALL_FACTOR = 0.9; // Text kleiner als 90 % des Fließtexts = Bildunterschrift/Fußnote → weg
// Abschnitte ohne Inhalt für Fragen (Quellenlisten, Links, Inhaltsverzeichnis)
const SKIP_SECTIONS = ["Inhaltsverzeichnis", "Einzelnachweise", "Weblinks", "Literatur", "Siehe auch", "Anmerkungen", "Quellen"];

// ---- 1. PDF → Zeilen -------------------------------------------------------
// pdf.js liefert pro Seite viele kleine Textstücke ("items") mit Position.
// transform[4] = x, transform[5] = y, item.height ≈ Schriftgröße.
// Stücke mit (fast) gleichem y gehören zur selben Zeile.
async function readLines(path) {
  const data = new Uint8Array(readFileSync(path));
  const pdf = await getDocument({ data, verbosity: 0 }).promise;
  const lines = [];

  for (let pageNo = 1; pageNo <= pdf.numPages; pageNo++) {
    const page = await pdf.getPage(pageNo);
    const { items } = await page.getTextContent();

    const rows = [];
    for (const item of items) {
      if (!("str" in item) || item.str === "") continue;
      const y = item.transform[5];
      let row = rows.find((r) => Math.abs(r.y - y) < 2);
      if (!row) rows.push((row = { y, items: [] }));
      row.items.push(item);
    }

    rows.sort((a, b) => b.y - a.y); // PDF-y zählt von unten → oben zuerst
    rows.forEach((row, i) => {
      row.items.sort((a, b) => a.transform[4] - b.transform[4]);
      let text = "";
      let prevEnd = null;
      for (const it of row.items) {
        const x = it.transform[4];
        // Lücke zwischen zwei Stücken → Leerzeichen, sonst direkt anhängen
        if (prevEnd !== null && x - prevEnd > 1 && !text.endsWith(" ")) text += " ";
        text += it.str;
        prevEnd = x + it.width;
      }
      text = text.replace(/\s+/g, " ").trim();
      if (!text) return;
      const size = Math.max(...row.items.map((it) => it.height || Math.abs(it.transform[3])));
      lines.push({ page: pageNo, text, size: Math.round(size * 2) / 2, posOnPage: i, rowsOnPage: rows.length });
    });
  }
  return { lines, numPages: pdf.numPages };
}

// ---- 2. Kopf-/Fußzeilen entfernen -----------------------------------------
// Was oben/unten auf vielen Seiten gleich steht (Ziffern ignoriert), ist
// Layout, kein Inhalt. Würde sonst in jedem Chunk die Suche verfälschen.
function dropRepeating(lines, numPages) {
  const norm = (t) => t.replace(/\d+/g, "#").toLowerCase();
  const isEdge = (l) => l.posOnPage < 2 || l.posOnPage >= l.rowsOnPage - 2;
  const pagesWith = new Map();
  for (const l of lines.filter(isEdge)) {
    const k = norm(l.text);
    if (!pagesWith.has(k)) pagesWith.set(k, new Set());
    pagesWith.get(k).add(l.page);
  }
  return lines.filter((l) => {
    if (/^(seite|page)?\s*\d+(\s*(von|of|\/)\s*\d+)?$/i.test(l.text)) return false;
    if (numPages >= 3 && isEdge(l) && pagesWith.get(norm(l.text)).size >= numPages * 0.5) return false;
    return true;
  });
}

// ---- 3. Überschriften erkennen → Abschnitte -------------------------------
function bodyFontSize(lines) {
  // Die Schriftgröße mit den meisten Zeichen ist der Fließtext.
  const chars = new Map();
  for (const l of lines) chars.set(l.size, (chars.get(l.size) ?? 0) + l.text.length);
  return [...chars.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
}

function toSections(allLines, docTitle) {
  let lines = allLines;
  const body = bodyFontSize(allLines);
  lines = lines.filter((l) => l.size >= body * SMALL_FACTOR);
  const isHeading = (l) =>
    l.size >= body * HEADING_FACTOR && l.text.length <= MAX_HEADING_LEN && /\p{L}/u.test(l.text);

  let sections = [];
  let current = { heading: docTitle, page: 1, parts: [] };
  let prevWasHeading = false;

  for (const l of lines) {
    if (isHeading(l)) {
      if (prevWasHeading && current.parts.length === 0 && l.page === current.page) {
        current.heading += " " + l.text; // mehrzeilige Überschrift
      } else {
        if (current.parts.length) sections.push(current);
        current = { heading: l.text, page: l.page, parts: [] };
      }
      prevWasHeading = true;
    } else {
      current.parts.push({ text: l.text, page: l.page });
      prevWasHeading = false;
    }
  }
  if (current.parts.length) sections.push(current);
  sections = sections.filter((s) => !SKIP_SECTIONS.includes(s.heading));

  // Abschnittsnummer: aus der Überschrift lesen ("3.2 Urlaub", "§ 5 ...") oder durchzählen.
  // "19. Jahrhundert" ist KEINE Nummer → einzelne Zahl mit Punkt wird nicht als ID gewertet.
  sections.forEach((s, i) => {
    const m = s.heading.match(/^(?<id>§\s*\d+(?:\.\d+)*[a-z]?|\d+(?:\.\d+)+|\d+(?=\s))\s+(?<title>.+)$/);
    s.sectionIdFromDoc = Boolean(m);
    s.sectionId = m ? m.groups.id.replace(/\s+/g, "") : String(i + 1);
    s.title = m ? m.groups.title : s.heading;
  });
  return { sections, body };
}

// ---- 4. Abschnitte → Chunks ------------------------------------------------
function joinLines(parts) {
  // Silbentrennung am Zeilenende rückgängig machen: "Pflege-\nkräfte" → "Pflegekräfte"
  let text = "";
  for (const p of parts) {
    if (/\p{Ll}-$/u.test(text) && /^\p{Ll}/u.test(p.text)) text = text.slice(0, -1) + p.text;
    else text += (text ? " " : "") + p.text;
  }
  return text;
}

function splitSentences(parts) {
  // Sätze mit Seitenzahl, damit jeder Chunk weiß, auf welcher Seite er beginnt.
  const out = [];
  for (const p of parts) {
    const sentences = joinLines([p]).split(/(?<=[.!?:])\s+(?=[\p{Lu}0-9„"•–-])/u);
    sentences.forEach((s, i) => {
      // Fortsetzung des Satzes von der vorigen Zeile → anhängen
      if (i === 0 && out.length && !/[.!?:]$/.test(out.at(-1).text)) {
        out.at(-1).text = joinLines([{ text: out.at(-1).text }, { text: s }]);
      } else out.push({ text: s, page: p.page });
    });
  }
  return out;
}

function toChunks(sections) {
  const chunks = [];
  for (const s of sections) {
    const prefix = s.sectionIdFromDoc ? `${s.sectionId} ${s.title} — ` : `${s.title} — `;
    let buf = { text: "", page: null };
    const flush = () => {
      if (!buf.text) return;
      const last = chunks.at(-1);
      if (buf.text.length < MIN_CHARS && last?.section === s.title) {
        last.chunk_text += " " + buf.text; // zu kurzer Rest → an Vorgänger hängen
      } else {
        chunks.push({ page: buf.page, section: s.title, section_id: s.sectionId, chunk_text: prefix + buf.text });
      }
      buf = { text: "", page: null };
    };

    for (const sent of splitSentences(s.parts)) {
      // Überlange "Sätze" (z. B. Listen ohne Punkt) hart an Wortgrenzen teilen
      const pieces = sent.text.length > MAX_CHARS ? sent.text.match(new RegExp(`.{1,${MAX_CHARS}}(\\s|$)`, "g")) : [sent.text];
      for (const piece of pieces) {
        if (buf.text && buf.text.length + piece.length > MAX_CHARS) flush();
        if (buf.page === null) buf.page = sent.page;
        buf.text += (buf.text ? " " : "") + piece.trim();
      }
    }
    flush();
  }
  return chunks;
}

// ---- 5. main ---------------------------------------------------------------
function slug(name) {
  return name.replace(/\.pdf$/i, "").toLowerCase()
    .replace(/[äöüß]/g, (c) => ({ ä: "ae", ö: "oe", ü: "ue", ß: "ss" })[c])
    .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
}

async function main() {
  const args = process.argv.slice(2);
  const outIdx = args.indexOf("--out");
  const out = outIdx >= 0 ? args[outIdx + 1] : "data/my-corpus.json";
  const pdfs = args.filter((a, i) => a.toLowerCase().endsWith(".pdf") && (outIdx < 0 || i !== outIdx + 1));
  if (!pdfs.length) {
    console.error("Usage: npm run chunk -- datei1.pdf datei2.pdf [--out data/my-corpus.json]");
    process.exit(1);
  }

  const allChunks = [];
  for (const path of pdfs) {
    const file = basename(path);
    const { lines, numPages } = await readLines(path);
    if (!lines.length) throw new Error(`${file}: kein Text gefunden. Gescanntes PDF? Dann braucht es erst OCR.`);

    const clean = dropRepeating(lines, numPages);
    const { sections, body } = toSections(clean, file.replace(/\.pdf$/i, ""));
    const chunks = toChunks(sections);

    chunks.forEach((c, i) => {
      allChunks.push({
        id: `${slug(file)}-c${String(i + 1).padStart(3, "0")}`,
        doc: file, // Quelldatei
        chunk_index: i + 1, // Chunk-Nummer innerhalb der Datei
        ...c, // page, section (Überschrift), section_id, chunk_text
      });
    });

    console.log(`${file}: ${numPages} Seiten · Fließtext ${body}pt · ${sections.length} Abschnitte · ${chunks.length} Chunks`);
    if (sections.length <= 1) console.warn(`  ⚠ Keine Überschriften erkannt. HEADING_FACTOR senken oder PDF prüfen.`);
  }

  const corpus = {
    doc: pdfs.map((p) => basename(p)).join(" + "),
    version: new Date().toISOString().slice(0, 10),
    chunks: allChunks,
  };
  writeFileSync(out, JSON.stringify(corpus, null, 2) + "\n", "utf8");
  console.log(`\n✓ ${allChunks.length} Chunks → ${out}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
