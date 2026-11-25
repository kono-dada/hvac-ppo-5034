// How the final report uses the figures: import a module and call `render(title: false)`
// inside a normal Typst `figure`, so the caption comes from the report.
// Build:  tools/typst/typst compile --root docs/report/typst --font-path tools/fonts \
//           docs/report/typst/figures-preview.typ docs/report/typst/build/figures-preview.pdf

#import "figures/env-schematic.typ" as schematic
#import "figures/env-step-response.typ" as step
#import "figures/env-days.typ" as days
#import "figures/env-controllers.typ" as controllers
#import "figures/method-loop.typ" as loop
#import "figures/ablation-learning-curves.typ" as curves
#import "figures/ablation-final.typ" as final
#import "figures/ablation-tradeoff.typ" as tradeoff
#import "figures/ablation-update-size.typ" as update

#set page(paper: "a4", margin: 2.5cm)
#set text(font: "New Computer Modern", size: 10pt)
#show figure.caption: set text(size: 9pt)
#show figure: set block(breakable: false, below: 2em)

#let fig(body, caption, label) = [#figure(body, caption: caption, kind: image) #label]

= Figure preview

#fig(schematic.render(title: false), [The room modelled as two heat stores.], <fig-schematic>)
#fig(step.render(title: false), [Two hours of full heating: the air reacts quickly, the walls slowly.], <fig-step>)
#fig(days.render(title: false), [Twelve randomly drawn days.], <fig-days>)
#fig(controllers.render(title: false), [HVAC off, a thermostat and the oracle on a cold and a hot day.], <fig-controllers>)
#fig(loop.render(title: false), [One round of PPO training.], <fig-loop>)
#fig(curves.render(title: false), [Learning curves of the ablation.], <fig-curves>)
#fig(final.render(title: false), [Results at the end of training.], <fig-final>)
#fig(tradeoff.render(title: false), [Energy versus discomfort.], <fig-tradeoff>)
#fig(update.render(title: false), [Size of each update.], <fig-update>)
