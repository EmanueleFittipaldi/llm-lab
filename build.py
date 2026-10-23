#!/usr/bin/env python3
"""Bundle src/ into a single self-contained dist/index.html (no build tooling needed)."""
import pathlib

root = pathlib.Path(__file__).parent
src = root / "src"
order = ["core.js", "m0_overview.js", "m1_tokenizer.js", "m2_embeddings.js", "m3_positions.js",
         "m4_attention.js", "m5_heads.js", "m6_block.js", "m7_sampling.js", "m8_lm.js", "m8_training.js", "m9_inference.js", "m10_posttraining.js", "m11_moe.js", "m12_scaling.js", "m13_quant.js", "m14_interp.js", "m15_rag.js", "m16_gpt.js", "main.js"]
js = "\n".join((src / "js" / f).read_text(encoding="utf-8") for f in order if (src / "js" / f).exists())
html = (src / "index.html").read_text(encoding="utf-8")
html = html.replace("/*__CSS__*/", (src / "style.css").read_text(encoding="utf-8"))
html = html.replace("/*__JS__*/", js.replace("</script", "<\\/script"))
out = root / "dist" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
# GitHub Pages serves the docs/ folder
docs = root / "docs"
docs.mkdir(exist_ok=True)
(docs / "index.html").write_text(html, encoding="utf-8")
(docs / ".nojekyll").write_text("", encoding="utf-8")
print(f"built {out} ({len(html)/1024:.0f} KB) and docs/index.html")
