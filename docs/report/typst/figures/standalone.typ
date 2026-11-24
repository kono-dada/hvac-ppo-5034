// Renders one figure on its own page, for preview and for including as an image.
//   typst compile --root docs/report/typst --input fig=env-step-response \
//     docs/report/typst/figures/standalone.typ out.pdf
// The report itself can instead import a figure module and place `render()` directly.

#let name = sys.inputs.at("fig", default: "env-step-response")
#import (name + ".typ"): render

#set page(width: auto, height: auto, margin: 8pt, fill: white)
#render()
