interface MarkdownProps {
  text: string;
}

export default function Markdown({ text }: MarkdownProps) {
  const segments: { code: boolean; content: string }[] = [];
  const codeRegex = /```(\w*)\n?([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = codeRegex.exec(text)) !== null) {
    if (m.index > last) {
      segments.push({ code: false, content: text.slice(last, m.index) });
    }
    const lang = m[1] ? ` class="language-${esc(m[1])}"` : "";
    segments.push({
      code: true,
      content: `<pre><code${lang}>${esc(m[2])}</code></pre>`,
    });
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    segments.push({ code: false, content: text.slice(last) });
  }

  const parts = segments.map((s) => (s.code ? s.content : block(s.content)));
  return <div className="text" dangerouslySetInnerHTML={{ __html: parts.join("") }} />;
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(s: string): string {
  const tags: string[] = [];
  const clean = s.replace(/<[^>]+>/g, (m) => {
    tags.push(m);
    return `\x00${tags.length - 1}\x00`;
  });
  const result = clean
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>")
    .replace(/(?<!_)_([^_]+)_(?!_)/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/~~(.+?)~~/g, "<del>$1</del>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  return result.replace(/\x00(\d+)\x00/g, (_, i) => tags[+i]);
}

function block(text: string): string {
  const lines = text.split("\n");
  const out: string[] = [];
  let inP = false;
  let listType: "ul" | "ol" | null = null;
  let items: string[] = [];

  function closeP() {
    if (inP) { out.push("</p>"); inP = false; }
  }

  function closeList() {
    if (listType) {
      for (const it of items) out.push(`<li>${inline(it)}</li>`);
      out.push(`</${listType}>`);
      listType = null;
      items = [];
    }
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "") {
      if (listType) {
        for (const it of items) out.push(`<li>${inline(it)}</li>`);
        items = [];
      } else {
        closeP();
      }
      continue;
    }

    if (trimmed.startsWith("<")) {
      closeP(); closeList();
      out.push(trimmed);
      continue;
    }

    const hd = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (hd) {
      closeP(); closeList();
      out.push(`<h${hd[1].length}>${inline(hd[2])}</h${hd[1].length}>`);
      continue;
    }

    const ul = trimmed.match(/^[-*+]\s+(.+)$/);
    if (ul) {
      closeP();
      if (listType !== "ul") { closeList(); listType = "ul"; }
      items.push(ul[1]);
      continue;
    }

    const ol = trimmed.match(/^\d+\.\s+(.+)$/);
    if (ol) {
      closeP();
      if (listType !== "ol") { closeList(); listType = "ol"; }
      items.push(ol[1]);
      continue;
    }

    closeList();
    if (!inP) { out.push("<p>"); inP = true; } else out.push("<br />");
    out.push(inline(line));
  }

  closeP();
  closeList();
  return out.join(" ");
}
