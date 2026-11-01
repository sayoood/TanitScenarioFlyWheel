import Prism from "prismjs";
import { useMemo } from "react";

// prismjs ships markup (xml), css, clike and javascript in its core; add JSON and the OpenSCENARIO DSL here
// instead of importing prism-components (those expect a global `Prism`, which bundlers do not provide).
Prism.languages.json = {
  property: { pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?=\s*:)/, lookbehind: true, greedy: true },
  string: { pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?!\s*:)/, lookbehind: true, greedy: true },
  number: /-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/i,
  punctuation: /[{}[\],]/,
  operator: /:/,
  boolean: /\b(?:false|true)\b/,
  null: { pattern: /\bnull\b/, alias: "keyword" },
};

Prism.languages.osc2 = {
  comment: /#.*/,
  string: { pattern: /"(?:\\.|[^"\\\r\n])*"/, greedy: true },
  keyword:
    /\b(?:import|scenario|action|modifier|extend|do|serial|parallel|one_of|with|wait|emit|call|until|def|struct|enum|global|type|param|keep|cover|record|if|else|event|elapsed|rise|fall|every|on|is|inherits|actor|it|true|false|path)\b/,
  "class-name": /\b(?:car|van|truck|bus|motorcycle|pedestrian|Path|signalized_junction)\b/,
  function: /\b[a-z_][a-z0-9_]*(?=\()/i,
  number: /\b\d+(?:\.\d+)?(?:mpsps|mps|kph|rad|deg|s|m)?\b/,
  operator: /[=<>!]=?|[+\-*/]|:|\./,
  punctuation: /[()[\],]/,
};

export type CodeLang = "xml" | "json" | "osc2" | "text";

function highlight(code: string, lang: CodeLang): string {
  const grammar = lang === "xml" ? Prism.languages.markup : lang === "text" ? undefined : Prism.languages[lang];
  if (!grammar) return code.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return Prism.highlight(code, grammar, lang);
}

export function CodeBlock({ code, lang, maxHeight = 460, label }: { code: string; lang: CodeLang; maxHeight?: number | string; label: string }) {
  const html = useMemo(() => highlight(code, lang), [code, lang]);
  const n = useMemo(() => code.split("\n").length, [code]);
  const gutter = useMemo(() => Array.from({ length: n }, (_, i) => i + 1).join("\n"), [n]);
  return (
    <div className="code-wrap overflow-auto rounded-xl border border-line bg-[#05080f]" style={{ maxHeight }} role="region" aria-label={label} tabIndex={0}>
      <div className="flex min-w-max">
        <pre aria-hidden className="m-0 shrink-0 select-none border-r border-line px-3 py-3 text-right text-fg-3">
          {gutter}
        </pre>
        <pre className="m-0 px-4 py-3 text-fg-1">
          <code dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
      </div>
    </div>
  );
}
