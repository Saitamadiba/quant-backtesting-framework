# How I Try To Fool Myself

### A field guide to the traps in systematic-strategy research

Most of what I have built in this repository exists to answer one question about a
trading idea: **is this a real edge, or am I the one being fooled?** Over roughly
sixty investigations, the overwhelming majority of my own candidates were
**refuted** — and a meaningful number of them were refuted only *after* they had
already convinced me once.

This document is the residue of those failures. Each entry is a trap I walked
into, the mechanism that made it convincing, and the rule I now apply so it
cannot happen the same way twice.

It is written for the researcher I was two years ago, and it is deliberately
about **method, not alpha**. A refuted hypothesis leaks nothing — by definition
it has no edge to protect — which is exactly why the failures are the part I can
share in full.

---

## How to read the numbers here

Statistics are quoted as **shapes**: t-statistics, R-multiple deltas, win rates,
year-by-year consistency, family-wise p-values. Instrument-and-parameter pairs
are withheld, because those are the only part that constitutes a recipe. Where a
result is stated as "closed" or "the effect vanished", that is the literal
outcome. Where a candidate survived, I simply don't discuss it.

A note on units: **R** is one unit of risk — the distance from entry to stop.
Expressing results in R rather than currency makes them sizing-agnostic, which
matters for reasons that become the subject of §4.

---

## 1. Leakage: the clock is the enemy

Every leak I have found reduced to the same thing — a number that was available
to my *backtest* at a moment when it would not have been available to my
*execution*. It is never announced. It arrives as an unusually clean result.

**1.1 — Timestamp every cross-sectional feature.**
I tested a classic divergence idea: one instrument makes a new extreme, a
correlated partner fails to confirm, trade the laggard. It scored **+0.0708R**
per trade. Then I changed exactly one thing — the partner's extreme was computed
from its *running* high at my confirmation bar, rather than from the whole
session's high. The effect fell to **+0.0002R**. Nothing else moved.

The "divergence" was hindsight wearing a costume. Read at the moment I would
actually trade, the partner had not yet failed to confirm; it merely failed to
confirm *later*. **Rule: a cross-sectional feature needs a timestamp as much as
a price does.** If you cannot say what the comparison instrument looked like at
your decision bar, you have not built a feature — you have built a memory.

**1.2 — Levels derived from running extremes invite same-bar look-ahead.**
Any level computed from a rolling max/min can quietly become known *within* the
bar that creates it. In one study a same-bar fill of this kind printed
**+32.6% per year**. The tell is that the entry is suspiciously often at the
best available price of the bar.

**Rule: audit every intrabar fill whose price derives from a running extreme.** A
useful implementation check: assert that the window used to build a level ends
strictly before the bar that may trade against it. In one parity self-check, this
class of error surfaced as **457 mass mismatches** between a rolling computation
and a window-current extreme.

**1.3 — Conditioning on the realised path falsely confirms.**
"Trades that reached +1R before drawing down performed well" is not a finding, it
is a restatement. Filters that inspect what the trade *did* cannot be entry
gates. **Rule: label every feature PRE-fill or POST-fill, in the schema itself,
and never let a POST-fill column appear in an entry study.** I now carry that
label in the database column names, because a naming convention survives
attention lapses in a way that discipline does not.

**1.4 — Never sort outcome deciles by (value, outcome) tuples.**
Sorting rows by a tuple whose second element is the outcome silently orders ties
by profitability, manufacturing a monotone relationship out of noise. It produces
beautiful dose-response curves from random data.

**1.5 — A break timestamp is a decision, not an observation.**
One strategy's apparent edge lived entirely in when I stamped a structural break.
On an honest clock — the break known only after the confirming bar closed — the
edge became **−2 to −9 basis points over n≈9,928**. The strategy did not change;
my bookkeeping did.

---

## 2. Controls: the cheapest thing that mimics the trade without the signal

This is the single highest-yield habit in the document. Before reaching for a
family-wise error correction, build the dumbest possible control that takes the
*same shape of risk* at the *same times* without using your signal. Most
candidates die here, and they die cheaply.

**2.1 — Pair every significance bar with a direct control.**
A cross-sectional momentum ranking looked strong until the control was "just buy
the highest-volatility instrument." Momentum content collapsed to **t = 0.68**.
The crown belonged to volatility; momentum was wearing it.

**2.2 — The placebo grid is the study.**
A calendar effect scored **t = +7.07** — overwhelming, on its face. I then
generated fake event grids with the same count and spacing but no real events.
**23 of them scored higher**, giving an honest **p = 0.609**. The scheduled event
was not the cause; the clock was. In that market, one particular hour carries a
session open, a funding stamp, and an expiry in the same minute — three suspects
in one room, and the calendar was merely the one I had a name for.

**Rule: when your effect is tied to a schedule, the null must be other
schedules**, not shuffled returns.

**2.3 — The random-bar drift control.**
For any bracketed trade (fixed stop, fixed target), ask what the *same bracket*
entered at a random bar would have earned. In one volume-pattern study, the
random-bar bracket scored the same as the signal, leaving excess **t = +0.15**.
The pattern was not selecting anything; the bracket was doing all the work.

**2.4 — Bracket asymmetry on a driftless path is not alpha.**
A stop-and-target pair on a random walk has asymmetric hit probabilities by pure
first-passage geometry. This reliably produces small positive expectancies that
look like edge. **Rule: re-run at the mirrored bracket.** If the "edge" survives
sign-flipping, it is geometry, not prediction. One inversion study died exactly
here: the measured asymmetry was **~2.9 bps**, comfortably less than a single
spread crossing.

**2.5 — A generic control often eats half the effect.**
A bearish volume signature cleared a family-wise bar convincingly. A generic
"any up-bar" control captured half of it, and adding the volume condition
contributed **+0.002R**. **A family-wise pass certifies the pattern; it says
nothing about the mechanism you attribute it to.**

**2.6 — Two components may be substitutes, not complements.**
A regime gate plus a ranking model appeared to work together. Running
equal-weight-plus-gate showed the ranking added **+0.003 Sharpe**. The gate was
the whole strategy. **One-step test for any "ranking + trend filter" design: drop
the ranking entirely and keep the filter.** If the result barely moves, you have
one idea, not two.

**2.7 — Drawdown-matching is the control most event studies omit.**
Matching control observations on realised drawdown, rather than only on date,
eliminated **68%** of an event-study effect. Events cluster in stressed
conditions; if the control does not share the stress, you are measuring the
stress.

**2.8 — Measure the overlap before proposing an extension.**
I once designed an enhancement to anchor signals to structural shelves, then
measured how many signals were *already* shelf-anchored: **96.2%**. The
enhancement was a no-op with a 3.8% tail. **Rule: quantify the overlap between
your proposed condition and the existing behaviour before building anything.**

---

## 3. The toll: decompose gross before any verdict

Execution costs are not a haircut applied at the end. They set a hurdle the raw
signal must clear *before* the idea is worth engineering at all.

**3.1 — Compute the fee-in-R wall a priori.**
Costs scale with trade frequency, but the stop distance scales with volatility,
so the *same* round-trip fee is a different fraction of R at each timeframe. For
one mechanism the round-trip toll worked out to roughly **0.24R / 0.53R / 1.03R**
at hourly / 15-minute / 5-minute bars. The signal survived scaling down; the
economics did not. **There is a timeframe below which a given fee tier cannot
support a mechanism, and you can compute it before writing any code.**

**3.2 — An entry must earn gross above the toll, or execution cannot save it.**
I have repeatedly been tempted by "the entry is fine, we just need better fills."
Sometimes true, usually not. In one book the gross edge was **+0.0647R** against
a toll of **0.1919R** — the toll was **3× the gross**. No fill improvement closes
that; the boat costs more than the cargo.

**3.3 — Beware overlays that are almost entirely denominator.**
A batch of cheap overlay filters showed apparent improvements that turned out to
be **88–100% attributable to the cost denominator** rather than to any change in
gross performance. Removing trades improves average net cost per trade whether or
not the removed trades were bad.

**3.4 — Some candidates have no gross to erode.**
Before optimising execution, check that gross is positive at zero cost. Several
ideas I pursued were negative *before* fees. A zero-cost backtest is the cheapest
diagnostic in the toolkit and I now run it first, always.

**3.5 — A stop on an always-on position can only ever be a fee.**
This one is nearly a theorem and it cost me real money to learn. If a book is
structurally always-on — it re-enters the same exposure on the next cycle — then
a protective stop does not reduce exposure, it merely round-trips it. In the case
I hit, the stop **paid 4.6× the loss it prevented**, and it *ratcheted*: a
trending underlying triggered it again on each subsequent cycle. Working the
algebra through afterwards, the toll-neutral trigger distance was about **14
standard deviations**, not the 3 I had set — the threshold was never going to be
reachable on a sane path. **Rule: a stop is only a stop if the flat state
persists.** Otherwise it is a subscription.

**3.6 — Premium quoted in volatility points is not premium quoted in currency.**
A variance-premium relationship that was clearly positive in volatility points
was far less compelling once converted to dollars, because the conversion is
state-dependent. **Rule: state every edge in the unit you actually get paid in.**

---

## 4. Denominators: what is "risk" measured against?

More of my sign errors came from this section than from any other. Two analyses
of the same trades can disagree in *direction* purely through what sits in the
denominator.

**4.1 — Always re-express a stop change at constant account risk.**
Tightening a stop mechanically improves R-multiples, because R itself shrank. It
looks like better trade selection; it is a change of ruler. **Rule: any
stop-distance comparison must hold account risk fixed** — which means position
size moves inversely. I rejected an otherwise attractive "tighter stop" result on
exactly this basis.

**4.2 — Price-basis and risk-basis results can disagree in sign.**
The same trade set was positive in basis points of price and negative in R, and
the cause was Jensen's inequality: constant-notional sizing and constant-risk
sizing weight the same outcomes differently. Neither is wrong; they answer
different questions. **Rule: name your sizing convention before quoting a
result**, and never mix the two in one table.

**4.3 — A growing notional is out-borrowing, not out-earning.**
I published, internally, that a constant-notional variant beat a compounding one
— then retracted it. A notional that grows with equity earns more currency
because it takes more risk, not because it has more edge. Comparing the two on
absolute return is comparing leverage.

**4.4 — Watch for two dials measuring different equity.**
In one live book, position sizing read the raw sum of a trade table while the
risk guard graded a separately computed virtual equity. The two drifted until the
bot was risking roughly **8.6% of the book per trade against a 1% target** — a
single full-stop loss would have been most of a month's budget. Both components
were individually correct in isolation. **Rule: sizing and risk enforcement must
consume the same equity function — literally the same function, not two
implementations of the same intent.**

**4.5 — A tighter stop is not a smaller loss when the rule is a percentage.**
Where an external rule is expressed as a percentage of *initial balance*,
tightening stops does not reduce the dollar risk that matters to that rule; it
only changes the R-denominator. Check whether the constraint you are optimising
against is denominated in your units or someone else's.

---

## 5. Multiplicity and effective sample

**5.1 — Family-wise bars must use a signed statistic where direction matters.**
Using max-|t| for a directional hypothesis inflated my own null by roughly **3×**
— it credited the null with extreme excursions in the direction I was not
claiming, making my bar far too easy. **Rule: signed max-t for directional
claims.**

**5.2 — Clustered books need an effective-n haircut.**
Simultaneous same-direction positions are one bet wearing several name tags.
Treating them as independent observations inflates t-statistics without adding
information. I apply day-clustered standard errors as the default, and note where
observation-weighted and cluster-weighted results **disagree in sign** — which
they sometimes do, and which is always worth a paragraph rather than a choice.

**5.3 — Date- versus observation-weighting can flip a conclusion.**
Weighting by observation lets high-activity days dominate. Weighting by date
treats a quiet day and a frantic one alike. Both are defensible; the trap is
computing one and reasoning about the other.

**5.4 — Survivorship is not a small correction.**
In one cross-sectional study, constructing the universe from currently-listed
instruments produced **+15.4 percentage points per year** of phantom performance
and **inverted the sign** of the conclusion. In an event study on delisted
tickers, **55% of events** involved instruments whose price history I could not
verify, and the direction of the bias could not even be signed — which made the
honest verdict "unanswerable with this data" rather than a number.

**Rule: when the bias cannot be signed, the result is not conservative — it is
unusable.** That is a harder standard than it sounds, and it has closed studies I
wanted to keep.

**5.5 — Trade floors are rate dials, not quality filters.**
This is the most recent entry in the collection. To stop an optimiser selecting
starved parameter sets, I raised the minimum in-sample trade count. It worked —
and it also silently mandated a trade frequency, because over a fixed scan window
a floor of *n* trades forces a rate of at least *n / window* by construction. My
floor mandated **≥7.22 trades/month**, and both instruments landed within **5%**
of exactly that. I had believed I was setting a quality standard; I was setting a
quota. **Rule: before imposing a sample-size floor, compute the frequency it
implies.**

**5.6 — A single-window re-fit cannot borrow the aggregate's tolerance.**
The same minimum-trade rule was safe in the original study, which averaged 72
windows — one starved window is diluted by seventy-one others. In a live tool
that fits *one* window and deploys its winner, the same rule is unguarded. **Same
rule, different exposure:** a panel of judges absorbs one eccentric member; handing
that member the gavel alone does not.

**5.7 — Win rate and expectancy are different objectives, and one is a trap.**
Trailing-stop overlays reliably **doubled win rate while losing R**. If you
optimise the metric that feels like being right, you will systematically trade
away the tail that pays for everything else. **Rule: state the objective
function before the experiment.** I have since had a case where a colleague's
argument for a wider target and mine for a tighter one were *both* correct —
because one of us was optimising expectancy and the other survivability. The
disagreement dissolved the moment the objective was named.

**5.8 — Effects can invert with sample size.**
One relationship measured **Spearman −0.257** on n=30 live fills and **+0.033**
on n=22,016. The small sample was not a noisy version of the large one; it was a
different, unrepresentative cohort. **Treat a live-sample result as a hypothesis
about the live sample, not a measurement of the mechanism.**

**5.9 — Your own optimiser's referee is worth listening to.**
I compute probability-of-backtest-overfitting and deflated Sharpe on every
walk-forward run. When these say **PBO ≈ 0.56** and deflated-Sharpe **p ≈ 0.94**,
the honest summary is "suggestive, not proven" — even when the mean looks good,
and even when I would rather write something stronger.

---

## 6. Is the instrument working?

A surprising fraction of my "results" turned out to be defects in the measuring
apparatus. These are the cheapest bugs to prevent and the most expensive to miss,
because a broken instrument reports confidently.

**6.1 — A threshold on a scale-dependent statistic needs per-scale calibration.**
A gate compared an indicator *spread* against a fixed threshold. The threshold
had been calibrated at hourly bars and was applied unchanged at 15-minute and
5-minute bars. But a spread shrinks with bar size while a constant does not — so
at the finer scales the gate admitted **essentially nothing**, and two detectors
recorded **zero signals for 31 days** while logging healthy cycles throughout. A
tide gauge held against ripples, reporting "flat."

The fix was to calibrate the threshold range per timeframe at a matched quantile.
The check that made it trustworthy: reading the method back at the *original*
timeframe reproduced the existing value to within 4% — so the method recovers the
known-good rung before I trust it on the unknown ones. The scaling turned out to
track **√time**, a random-walk signature, which the codebase had already encoded
for a *different* threshold and never applied to this one.

**Rule: inventory every threshold that compares against a price-derived spread,
and ask what happens to it under a change of sampling interval.**

**6.2 — A placeholder is not an outcome.**
A scoring pipeline stamped unscorable trades with a sentinel status at **0R**.
Downstream aggregation treated those zeros as *flat trades*. The book read
**−0.087R at a 6% win rate**; the truth over the scorable subset was **−0.2152R
at 15.7%**. Every individual component behaved as designed.

**Rule: a sentinel must be excluded by construction, not by convention** — filter
it in the query that defines the population, and assert the excluded count.

**6.3 — Verify the instrument is reading the tape you actually trade.**
A detector reported **+6.18R**. It was computing against a data feed that did not
match the one the live book executed on. Honest gross was **−0.032R**, and
mirror-symmetric — no edge in either direction.

**6.4 — Silence is not evidence of health.**
Several defects presented as *nothing happening*, which is indistinguishable from
"no signals today":
- A cooldown timer fired after **every** evaluation, including ones that found
  nothing, leaving the detector blind for hours at a stretch.
- A close-matching routine keyed on the *opening* order id where the venue stamps
  the *closing* one, so **zero** closes were ever booked, positions accumulated
  to the concurrency cap, and the book went inert for two days.
- A kill switch and a halt flag that were both verified, later, to be **inert** —
  placebo controls I had trusted for weeks.

**Rule: instrument the negative path.** Log *why* nothing happened, count it, and
alert on the absence of activity — not only on errors. A turnstile that reports
zero visitors and a turnstile beside a welded door produce identical readings.

**6.5 — A hang is a test result too.**
A test suite that appeared merely slow was in fact hanging forever: importing the
production module set an HTTP proxy process-wide to an address only routable
inside the deployment network, so any test doing real I/O blocked on a dead
letterbox. Once diagnosed, the whole suite ran in under two minutes. **Rule:
chase a hang, don't wait it out** — and never let a module set global network
state as an import side effect.

**6.6 — Know what your read-only reader cannot see.**
A database book appeared empty for a month. It was not: the reader I used cannot
see write-ahead-log content, so it reported a stale, empty view with no error. The
bot had been writing all along. **Rule: verify your read path against a known
write before concluding anything from an empty result.**

**6.7 — A gate that re-evaluates cannot filter the feature it froze.**
A distance filter was intended to reject far-away setups. Because it
re-evaluated every cycle while the setup rested for hours, it rejected **0 of 30**
live fills. **Rule: separate the decision to *place* from the decision to
*hold*** — they are different questions and a single threshold cannot answer both.

**6.8 — Never book a value that does not reconcile.**
A partial-close path recorded a profit equal to an entire short notional,
because a "position already flat" branch skipped the leg-recording call. The
phantom profit then logged a qualifying day and **reset the very circuit breaker
its own losses had just tripped**. **Rule: make reconciliation a precondition of
booking — fail closed if the legs don't balance.**

**6.9 — Measure a data source's base rate before mining it.**
Before extracting strategies from a public forum, I measured what fraction of
posts contained a falsifiable rule: about **15%**, and a single promotional
account dominated the corpus. That measurement changed the project's scope more
than any subsequent analysis. **Rule: characterise the source before trusting
anything derived from it.**

---

## 7. Process: pre-registration, and the discipline of retraction

**7.1 — Pre-register the expectation and the kill/keep bar before the forward
test.** Written down, dated, with the sample size at which it will be read. The
purpose is not ceremony; it is to deny my future self the freedom to relocate the
goalposts once data starts arriving. Every forward test in this program has a
frozen bar in a file, and the read happens at the stated n — not at the n where
it looks best.

**7.2 — Do not widen a band to fit the result.**
A configuration missed a pre-registered acceptance band by **2.9×**. I had a
plausible argument that the band's *reference value* was itself contaminated, so I
measured that argument — and it was **wrong**: the correction moved the reference
by 0.03 units, essentially nothing. The honest outcome was that my hypothesis
lost and the configuration was rejected. **Rule: you may re-derive a bar's
reference on stated grounds; you may not adjust the bar because a result missed
it.** Those feel similar in the moment and are opposites.

**7.3 — Retract explicitly, in the same place you claimed.**
I have retracted my own findings more than once — a sizing claim, an exit-overlay
valuation, a contamination hypothesis. Each retraction is written into the record
next to the original claim, because a correction filed somewhere else is not a
correction. The most useful entries in my research log are the ones that say *I
believed X, here is the measurement that killed it.*

**7.4 — Stamp era breaks.**
Any change to a live system's behaviour partitions its results. Without an
explicit era boundary, a fix's improvement gets averaged with the broken period's
damage and both become unreadable. Related: a permanent halt is a **latch** — it
persists across restarts by design, so an era reset must be sequenced *after* old
positions are flat, or the fresh book inherits the old book's exposure.

**7.5 — Check the log before starting work.**
I once completed an analysis that a parallel effort had finished eleven hours
earlier. **Rule: read the index for the specific item you are about to start**,
not just the general area.

---

## What this record is worth

Three claims, and I would defend each of them in an interview:

1. **The negative results are the asset.** A library of closed questions means
   the next candidate that touches a refuted design space must state which
   closure it is challenging and what *new* mechanism justifies re-opening it.
   That is compounding, and it is cheaper than re-running dead ends.

2. **Most of the value is in the controls, not the corrections.** Family-wise
   error control is necessary and insufficient. The cheap direct control — the
   random bar, the highest-volatility instrument, the fake calendar, the mirrored
   bracket — killed more of my candidates than any p-value adjustment, and killed
   them for reasons I could explain.

3. **The instrument deserves as much scrutiny as the hypothesis.** Zero signals
   for 31 days, a book that read wrong by 2.5×, a detector on the wrong feed, a
   stop that was purely a fee. None of these were strategy errors. All of them
   would have been reported as strategy results.

*Statistics quoted are shapes rather than recipes; parameters, live
configurations, and surviving candidates are deliberately withheld.*
