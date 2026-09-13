# Large Language Models in IVF Laboratory Analytics

### An interpretive layer over the IVF Digital Twin: how deterministic computation, predictive models and a hierarchy of Claude agents are assembled into one auditable analytical system

---

> **A note on the intended reader.** This document is written primarily for clinicians and embryologists rather than for engineers. It assumes no knowledge of how neural networks work internally, and it deliberately explains the properties of language models in operational terms — what they are reliable at, what they fail at, and what follows from that — rather than in mathematical ones. Where a technical mechanism matters to the trustworthiness of a clinical conclusion, it is explained in full; where it does not, it is left out. No clinic-specific data, performance figures or results appear anywhere in this text: the subject here is the *system* and why it is built the way it is.

---

## 1. Where this comes from, and what problem it solves

The IVF Digital Twin was built to answer a question that clinical embryology had never been able to answer quantitatively: *given everything we know about this couple and this cycle, what is the expected trajectory of this attempt?* It does this with an ensemble of models operating at different levels of abstraction — a Monte Carlo simulation of the biological funnel, a Bayesian posterior over that simulation, an attention-based tabular learner, a graph model that reasons over similarity between patients, a diffusion model for conditional imputation of missing stages, a foundation model for tabular prediction, and a final ensemble that fuses them. The output is a probability, a set of predicted stage counts, and an uncertainty band.

That output is genuinely useful and genuinely incomplete. A probability tells a clinician what to expect. It does not tell a laboratory *why* the expectation was not met. When a transfer fails, the question that actually gets asked in the morning meeting is not "what was the predicted probability" but "what happened in this cycle that we could have influenced" — and that question is not answered by any single model, because the answer lives in the space *between* sources: between the laboratory's own key performance indicators and the international consensus benchmarks, between what the model predicted and what the funnel actually delivered, between the culture window and the incubator's physical logs, between the embryo cohort and its genetic testing, between one embryologist's cycles and another's, between one consumable lot and the next.

Nobody had been putting those sources in one place, because putting them in one place is not a modelling problem. It is a problem of *reading* — of holding a dozen heterogeneous, partially missing, differently-denominated data sources in mind at once and producing a coherent, honest, non-overreaching account of a cohort. That is exactly the class of problem large language models are good at, and exactly the class of problem they are dangerous at, because a fluent model will produce a coherent account whether or not the evidence supports it.

This document describes the system we built to get the first without the second: an analytical pipeline in which **deterministic Python computes and classifies every number, and language models only explain what has already been computed** — with an explicit adversarial check on the explanation before it is allowed into the final report. It is an extension of the Digital Twin, not a replacement: the Digital Twin remains the predictive core, and everything described here is the reasoning layer wrapped around it.

---

## 2. What a language model can and cannot do with data

Before describing the system, it is worth being precise about the instrument, because almost every design decision in this pipeline is a direct consequence of a specific strength or a specific weakness of language models. Clinicians encountering these tools usually meet them as a chat window, which is the least informative possible introduction: it shows the fluency and hides the mechanics. The properties that matter for clinical analytics are not the ones a conversation reveals.

### What they are genuinely good at

**Reading heterogeneous material at once.** A language model can hold a laboratory's indicator table, an incubator log summary, a genetic testing summary, a stimulation-protocol distribution and a list of per-patient predictions in mind simultaneously and reason across them. This is not a small thing. It is precisely the operation that no database query performs, that no dashboard performs, and that a human performs only when given uninterrupted time — which in a working laboratory means almost never.

**Reconciling vocabulary.** The same clinical concept appears under different names in different systems, in different languages, in free text and in coded fields. Mapping between them is a task models do well, and it is the task that historically consumed most of the effort in any cross-source analysis.

**Explaining a computed result at the reader's level.** Given a number and its context, a model can produce an account of what that number means for this laboratory, in the register the reader needs — consultative for a physician, technical for a quality manager. This is the core competence being used here.

**Following explicit written rules, reliably, at the moment of use.** If the rule is present in front of the model when it works, it is followed with high consistency. This is the property the entire instruction layer of this project is built on.

**Doing all of the above identically the hundredth time.** A model does not get tired, does not skip the boring source, and does not read the fourth cohort less carefully than the first. Consistency over repetition is a genuine advantage over human reading, and it is what converts an occasional analytical exercise into a routine process.

### What they are bad at, and why

These are not defects to be fixed by a better prompt. They are structural properties of the technology, and a system that does not design around them will fail in ways that look like success.

**Fluency is not correctness, and the model cannot tell the difference from the inside.** A language model generates text that is a plausible continuation of what came before. Plausibility and truth coincide most of the time, which is exactly what makes the exceptions dangerous: a wrong statement is produced with the same confidence, the same vocabulary and the same fluency as a right one. There is no internal signal that separates "I know this" from "this sounds like the kind of thing that would be true here". A reader cannot detect the difference by tone, because there is no difference in tone.

**Arithmetic over many values is unreliable — and, more importantly, unauditable.** Even when a model computes a rate correctly, that computation cannot be reproduced, cannot be unit-tested, and may come out differently on the next run. In a clinical setting the reproducibility matters more than the accuracy: a number nobody can regenerate is a number nobody can defend.

**Confabulation under pressure to produce output.** Asked to find problems, a model will find problems. Asked for three causes, it will supply three. The instruction is treated as a description of the expected output shape, and an empty answer feels to the model like a failure to comply. Any review step that does not explicitly bless an empty result will manufacture findings.

**The pull of a good narrative.** When evidence is ambiguous, a model resolves the ambiguity in the direction the surrounding text implies. A well-formed story attracts supporting detail. This is the mechanism behind the most expensive failure mode in analytics of this kind — not a fabricated number, but a *real* number recruited into an argument it does not actually support, or the same measurement appearing twice under two names and being counted as two independent confirmations.

**Confident application of the general convention where the local one differs.** A model's default knowledge is the field's textbook consensus. Where a particular laboratory, export or indicator has a legitimate local peculiarity, the model will not hesitate — it will apply the standard reading. The canonical example in this domain is an indicator exceeding one hundred per cent, which any general reader will call corrupt data, and which in fact has specific benign explanations that are clinically informative. The general reading is not merely wrong; it destroys the signal and reports a data-quality problem that does not exist.

**Statistical blindness unless the denominator is placed in front of it.** A rate computed on a handful of cycles reads exactly like a rate computed on hundreds. Unless every number carries its numerator, denominator and interval, a model will treat them as equals — and so, reading the resulting text, will the clinician.

**No memory between sessions.** Nothing established in one working session persists into the next unless it was written to a file. A correction made in conversation is lost the moment the conversation ends.

**Inability to check its own work.** The same process that produced an error will, on review, endorse it — the error looked plausible the first time and looks plausible the second. Self-review by the same instance in the same context is close to worthless.

**Degradation with volume.** Given an enormous undifferentiated input, attention thins out and whole sections can be silently under-weighted. A model given everything reads less carefully than a model given exactly what it needs.

**Every disclosure is permanent.** Text placed in a model's context has left the local machine. In a domain governed by medical confidentiality this is not a policy question but an architectural one.

### The rules we derived, and what each one answers

The pipeline's design is a point-by-point response to the list above. It is worth reading these as paired with their reason, because each rule looks arbitrary until you see what it prevents.

**Every number is computed by code, never by a model.** *Answers: fluency without correctness; unreliable and unauditable arithmetic.* Agents receive computed values and may only speak about them.

**Classification is also done by code.** *Answers: the pull of the narrative.* The packet does not hand the model a raw rate and ask for a verdict; it hands over the verdict as well. When the packet says a result is moderate, the agent explains moderate — it is explicitly forbidden to re-grade a borderline case because the surrounding text made another grade feel more fitting.

**Agents receive small, pre-structured packets, never raw files.** *Answers: degradation with volume; permanence of disclosure.* Each role gets what it needs and nothing else, in a form already reduced to the relevant fields.

**What must not be said is written down as explicitly as what must be.** *Answers: confident application of the general convention.* The known misreadings of this domain are recorded by name, together with the correct interpretation, in the files the agents load at the moment of use.

**Fact and hypothesis are separated at the level of grammar.** *Answers: confabulation.* Causal statements are framed as reasoned hypotheses with a named basis. Correlated events are candidate confounders. The vocabulary of prescription is prohibited outright in the patient-facing role.

**Every number travels with its denominator.** *Answers: statistical blindness.* Numerator, denominator, interval, population definition and direction of control accompany every indicator, and the interpretation is required to name the denominator when two versions of the same concept appear.

**Absence is typed rather than assumed.** *Answers: narrative pull applied to gaps.* Zero is an observation. A logging gap is an event. An unmatched report is unmatched. A data source that was never configured is not evidence that the phenomenon did not occur — and a strong claim in either direction from an empty section is treated as a defect.

**The writer is never the checker.** *Answers: inability to self-verify; confabulation under pressure.* A separate instance, with the same inputs and a single adversarial question, reviews the finished interpretation before it can be published — and is told in writing that finding nothing is a valid and useful outcome.

**Persistence lives in files, not in the model.** *Answers: no memory between sessions.* Decisions, corrections and refuted conclusions are written into versioned documents that are loaded at the start of every session.

**Identity never enters a model's context.** *Answers: permanence of disclosure.* The analytical layer works on pseudonyms; identity is re-attached afterwards, by code.

**The expensive model is used only where judgement is genuinely required.** *Answers: economics, and indirectly consistency.* Bounded explanation over pre-classified input goes to a small fast model, which is not a compromise — it is the correct tool for a constrained task, and cheap enough to run on every cohort every time.

Everything that follows is an elaboration of these eleven rules.

## 3. The founding principle: computation and interpretation are separate jobs

The first of those eleven rules carries more weight than the other ten together, and the rest of this document is easier to read once it has been stated in its strongest form:

> **Python calculates and classifies. Agents only explain. No number is ever produced, recomputed, rounded, or "sanity-adjusted" inside a language model.**

This is not a stylistic preference. It is the reason the system can be trusted in a clinical setting. If a language model computes a rate, that rate is unauditable — it cannot be reproduced, it cannot be unit-tested, and it will drift silently between runs and between model versions. If a language model *reads* a rate that Python computed from a versioned contract, the number is reproducible forever, the model's contribution is confined to language, and any error the model makes is an error of interpretation — which can be caught by another reader.

The corollary is that classification is also Python's job, not the agent's. When the pipeline hands an agent a patient record, the record does not simply state a blastocyst yield as a percentage. It carries the yield, the numerator, the denominator, the confidence interval, the direction of control, the benchmark, and the pre-computed status label. The agent is instructed, in its own system prompt, not to re-classify: if the packet says *moderate*, the agent explains *moderate*. This removes an entire category of failure in which a model quietly re-grades a borderline case because the surrounding text made a different grade feel more narratively satisfying.

The second founding principle is a privacy boundary that is enforced structurally rather than by instruction, and it has its own section below.

The third is that **fact and hypothesis are always separated in language**. Every causal statement in the final report is a hypothesis with a named basis. A mid-window change of a consumable lot is a candidate confounder, never a cause and never a judgement about the supplier. A thermal excursion during the culture window is an adverse event that coincided with a cohort, not the reason the cohort failed. The system consults; it does not diagnose and it does not prescribe.

---

## 4. The shape of the pipeline

The whole run is a single sequence, triggered by one command in the terminal. Conceptually it has six layers, and each one feeds the next:

**Layer 1 — Ingestion and schema resolution.** Laboratory exports arrive as spreadsheets from a clinic information system, and no two clinics name their columns the same way. Column names are resolved *only* through an explicit configuration file of canonical names and aliases. Unknown column names are never guessed. If a required column cannot be resolved, the run stops rather than proceeding on an assumption. This sounds pedantic until you consider the alternative: a silently mismatched column produces a plausible report built on the wrong denominator, and nobody notices for months.

**Layer 2 — Normalisation and pseudonymisation.** Transfer outcomes are binarised from free-text clinical language; a small number of clinical variables are imputed by a fixed, documented rule. Two conventions matter enormously here and are frozen at the project level: **zero is an observation, not a missing value** — a cycle with no blastocysts is data, not absence of data — and only a genuinely empty cell means *unknown*. Then each cycle receives a locally-computed keyed hash identifier. From this point on, the analytical layer of the system speaks only in that pseudonym.

**Layer 3 — Laboratory KPI.** The laboratory's own performance indicators are computed against international consensus methodology: fertilisation, cleavage, blastocyst and good-blastocyst yields, degeneration and abnormal-fertilisation rates on the embryology side, and oocyte-retrieval and maturity rates on the clinical side. Every indicator carries a numerator, a denominator, a confidence interval, a direction of control (for most indicators a floor, for the two adverse-event indicators a ceiling), a competence level, a benchmark, and a status. Cohort-level rates are computed as *sum of numerators over sum of denominators* — never as the mean of individual cycle ratios, which is a classic and badly misleading error on small cohorts.

Two structural decisions in this layer carry disproportionate weight. First, embryology indicators and clinical indicators live in **separate tables that are never mixed, compared, or ranked against each other**, because they measure different people doing different jobs at different stages of the funnel. Second, laboratory KPI are computed on the **broad laboratory population** — every cycle that produced oocytes — and not on the narrower cohort of cycles that reached transfer with a known outcome. The reason is conceptual: laboratory KPI describe what the laboratory did with the oocytes it received, not how the transfer ended. If you compute them on the transfer cohort, you are silently describing a self-selected subset, and on a small export you can end up with an indicator resting on a handful of cycles while the report presents it as a laboratory-wide fact.

That broad population is then narrowed by three explicitly *nested* exclusion sets — cycles still in progress at export time, cycles with no retrieval, freeze-all cycles; then cycles whose funnel counters were carried over from a different retrieval; then cycles that never reached insemination but whose retrieval-stage numbers are a complete observation of clinical performance. Each of these exists because a real export produced a real distortion without it, and each is applied only to the indicator families it actually affects. This kind of thing is invisible in a methods section and decisive in practice.

**Layer 4 — Predictive models.** The Digital Twin ensemble, the tabular foundation model that produces a per-transfer pregnancy probability with a confidence interval, and — as an independent quantitative quality-assurance layer — a regression model that predicts the *counts* at each funnel stage as a distribution rather than a point, so that the actual result can be checked against the predicted interval. An optional independent neural network can be run as a cross-check, and is declared invalid unless the exact training-time scaler is loaded and the feature order matches; fitting a scaler at inference time is forbidden by contract, because a model that re-fits its own preprocessing is not an independent check of anything.

Each model runs in its own environment, launched by an adapter. Results are joined locally on the clinical identifier and immediately validated against versioned artefact contracts — required fields, key uniqueness, ordering of quantiles, freshness. A stale result file from a previous run does not count as a current result. A contract violation stops the run.

**Layer 5 — Forensics of the culture window.** For the cycles that warrant attention — those flagged by the analytics review criteria or those whose prediction interval is too wide to be actionable — the pipeline reconstructs the physical and biological context of the days between retrieval and transfer:

- *Incubator forensics.* The incubator's own log files are parsed into episodes: excursions from setpoint, their duration and maximum deviation, severity, recovery time, and gaps in logging. A gap is treated as an event, not as absence of data — a short gap means the ability to *write* logs was lost, which is itself an adverse occurrence, while a long gap means the incubator or its logging was deliberately stopped. These are different facts and the system labels them differently, precisely because the intuitive reading ("no data") is the wrong one.
- *Consumable lot forensics.* The laboratory's inventory system is queried live for the lot history of each consumable position, and the pipeline determines whether more than one lot period genuinely overlapped a given culture window. A mid-window change is recorded as a candidate confounder — a thing to be aware of when reading that cycle — and explicitly not as a cause or a quality judgement.
- *Genetic testing forensics.* Where preimplantation genetic testing reports exist for the cohort, they are parsed and summarised into cohort-level features: the euploidy rate, sequencing quality-control noise computed *only over the euploid subset* (as a proxy for the biopsy and sequencing quality of the embryos that were actually transferable), and whether analysis settings varied between embryos of the same report. Matching a report to a cycle is done by name plus a date window, and where that does not resolve unambiguously, no match is made and the reason is recorded — the system does not guess an identity.

**Layer 6 — Cycle-level arrest patterns and clinical context.** A separate module tests each cycle for developmental arrest patterns at defined transition points, using a one-sided exact binomial test of the observed proportion against a published threshold rather than a naive "below the line" comparison, and marks stages with too few observations as *insufficient data* rather than dropping them. Recurrence across a patient's cycles is a criterion for *discussing* genetic counselling — a conversation, not a diagnosis. In parallel, where the clinic information system carries them, descriptive clinical fields — stimulation protocol type, total gonadotropin dose, trigger, cycle days, diagnosis codes — are assembled into a purely descriptive layer that is deliberately kept out of the models' input path entirely, so that it can be used for interpretation without contaminating prediction.

Everything is written atomically — temporary file, then replace — together with a run manifest carrying schema versions, artefact hashes and the status of every soft step.

---

## 5. The Digital Twin and its narrator: a language model wired directly into a predictive model

The pattern described in this document did not begin at the cohort level. It began inside the Digital Twin itself, in the component that turns a prediction into something a physician can use — and that component is worth describing separately, because it is the clearest possible illustration of the whole principle in miniature.

### Why a prediction needs a narrator at all

The Digital Twin produces, for a single cycle, a set of numbers: a probability of pregnancy per transfer, an uncertainty interval around it, expected counts at each stage of the funnel, a cluster assignment, and a family of risk estimates. All of that is correct and almost none of it is directly usable in a consultation.

A physician sitting with a couple does not need a vector. They need to know: is this expectation favourable or poor *relative to what this clinic normally sees*; how confident is the estimate and is it confident enough to act on; which stage of the funnel is the limiting one; what in this picture is modifiable and what is not; and how to say all of that to a patient without either false reassurance or unwarranted alarm. Those are translation tasks, and they are exactly what a language model does well.

More importantly, they are tasks where getting the *register* wrong causes real harm. A probability presented flatly is heard as a promise. An uncertainty interval presented without framing is heard either as precision or as ignorance, depending on the listener. A model output described in the imperative — "administer", "perform", "indicated" — has crossed from decision support into prescription, which is not what any of this is for.

### How it is implemented

The narrator is built on a strict division that predates this pipeline and was inherited into it: **Python decides the categories, the language model produces the sentences.**

A deterministic module assembles, for each cycle, a compact *briefing packet* — a small structured object containing that cycle's values together with the categories already assigned by fixed, documented thresholds: whether the prognosis band is favourable, good, moderate or low; whether the statistical uncertainty is low, moderate or high; which funnel stages are within or outside expectation; which risk flags apply. There are no model calls anywhere in this module. Its entire job is to reduce a wide row of numbers into a small, already-judged brief.

Packets exist at three levels of resolution: a per-cycle narrative packet for the treating physician; an ensemble packet that exposes the layered structure of the prediction for anyone auditing how the estimate was formed; and a cohort packet carrying the laboratory-level context. The same principle applies at all three — the classification is code, the prose is the model.

The narrator role that consumes the per-cycle packets is the most tightly constrained agent in the system. It works in batches of eight to ten cycles per call, handling strictly and only the identifiers named in that call, with a fixed set of subsections per cycle and an absolute prohibition on numbers crossing between patients within a batch. It may not re-classify. It may not diagnose. Its permitted and forbidden verbs are enumerated.

### The uncertainty band, and why classification belongs in code

One detail from this component deserves to be told in full, because it demonstrates the entire argument better than any general statement could.

The width of the confidence interval around a probability is used to decide whether a prediction is stable enough to act on: a narrow interval means the estimate is well-determined, a wide one means it is not. The natural implementation is to compute the width and band it — narrow, moderate, wide.

The failure is that a *degenerate* interval — one that was never actually computed, and therefore has zero width — is indistinguishable from an infinitely precise one if you only look at the number. Read naively, an uncomputed interval becomes the strongest possible evidence of certainty. Cycles with no interval at all were being presented as the most reliable predictions in the cohort, and one of them received the highest available reliability grade for an interval that did not exist.

The fix has two parts, and both are instructive. First, a degenerate width is now classified as *interval not computed — not usable as a signal*, which is a different category from "narrow", not a value on the same scale. Second, the degeneracy test is a small threshold rather than an exact comparison with zero, because floating-point artefacts produced widths of a hundredth of a percentage point that passed a strict "greater than zero" test, were banded as low uncertainty, and were then rounded to zero only at display time — invisible in the output, decisive in the classification.

No amount of prompt engineering catches this. A language model handed the number would have read it exactly as the naive code did, and would have written a confident and completely wrong sentence about a well-determined prognosis. It was caught because the classification lives in code that can be read, tested and reasoned about — and the reason it will stay caught is that the explanation is now written down next to the threshold.

### From a local helper to a governed layer

In its original form, this narrator ran against a small model executing locally inside the prediction project. Moving it into the present system changed two things and preserved one.

What was preserved is the packet structure and the division of labour — the same deterministic classification, the same shape of brief.

What changed is that the model call was removed from the module entirely and moved into the agent layer, where it is subject to the privacy boundary, the role definitions and the review chain described in the rest of this document. And what the packets are consumed by is now a frontier model rather than a small local one, which materially improves the quality of the clinical prose without changing a single number, because no number was ever the model's to produce.

That is the general lesson of this section: **the interface between a predictive model and a language model should be a pre-classified brief, not raw output.** When it is, you can upgrade either side independently, and the classification remains auditable forever.

---

## 6. Connectors: how the pipeline reaches external data and external code

A laboratory's analytical picture is assembled from systems that were never designed to talk to each other. The pipeline therefore contains two distinct families of connector, and both are worth describing because they are where most real-world fragility lives.

### Connectors to data sources

**Shared cloud storage, for the incubator's own logs.** The authoritative copy of the incubator log archive lives in shared storage rather than on the analysis machine. The connector does not mirror the archive: it computes, from the current run's review cohort, exactly which culture windows matter, determines which weekly log folders intersect those windows, and downloads only those. This is a deliberate constraint — pulling an entire multi-year archive to analyse a handful of cycles is both slow and an unnecessary expansion of what sits on the local disk.

**A live query against the consumables inventory system.** Lot histories are pulled through the inventory platform's API at run time rather than from an exported snapshot, because a snapshot ages and a lot timeline that is one week stale can miss precisely the change you are looking for. Access is gated behind a token that never enters version control, and the entire forensic step is disabled by default for laboratories that do not have such a system.

**Browser automation, for the genetics portal.** Genetic testing reports are available only through a personal-account web interface with no API. The connector drives a real browser session interactively, locating the relevant cases for the current cohort and downloading finished reports. Parsing is then handled by a deterministic module, not by the browser session — the connector's only job is acquisition.

Two properties are common to all three and are more important than the mechanics.

*They are interactive rather than headless.* These steps run from within the working session, under supervision, because authentication is interactive, portal layouts change, and matching a downloaded report to a cycle occasionally requires judgement. A scheduled script that silently fails at any of these produces the worst possible outcome: a report that looks complete and is missing a data source.

*They are gated and soft.* A laboratory without an incubator log archive, without an inventory system, or without accessible genetic reports runs the same pipeline with those steps switched off, and the corresponding sections are simply absent. Critically, the system knows that *absent* and *not run* look identical downstream, and forbids any strong conclusion drawn from an empty section.

### Connectors to the code of other prediction systems

The second family is different in kind: these are adapters that reach into *other software projects* — separate codebases, separate Python environments, separate dependency stacks, sometimes separate hardware requirements — run them, and bring their output back under contract.

The mechanism is uniform. Each external tool is described by a command template with placeholders for its input file, its output file and the repository root; the adapter substitutes those, launches the tool in its own project directory with its own environment, waits for it to finish, and then validates what came back. Absolute machine paths live in exactly one local file, so moving to a different computer changes one file and nothing else.

The validation is where the engineering actually is, and it is built around a single principle: **a result file only counts if this run produced it.** The adapter records the launch time and rejects any output older than it; where the external tool emits a run identifier, the adapter checks that the identifier matches the one it expects; the returned table is validated against a versioned contract for required fields, key uniqueness and internal consistency (for instance, that predicted quantiles are properly ordered). Reading yesterday's result file and reporting it as today's is the classic silent failure of any multi-tool pipeline, and it is designed out rather than watched for.

Connecting to someone else's code also means reading it rather than trusting its filenames. One of the external tools writes two output variants, and the one whose name promises outcome-annotated results turned out to discard entire rows whose outcome was *negative*, because its text-parsing routine classified the word for "no" as an unknown value and dropped the record. Every report built on that file therefore described successes only — silently, with no error anywhere. The adapter now reads the other variant and reconstructs the outcome locally, the reasoning is documented at the point of the decision, and a test reproduces the external project's parsing verbatim so that the day it is fixed upstream, we find out.

### Why the same cohort is scored by more than one system

The multiplicity of prediction systems here is not redundancy for its own sake. Each one verifies a different axis, and the pipeline is explicit about which.

**The ensemble** is the primary estimate: multiple model families combined, each contributing a different inductive bias, with the final fusion as the headline probability.

**The tabular foundation model** contributes a per-transfer probability with an explicit uncertainty interval. It is trained under a completely different regime from the ensemble's other members, which is what makes it valuable. It is also, itself, one of the ensemble's inputs — and this is precisely why the project fixes the canonical names of all models in a written glossary. The same estimate appearing under two names in two sections of a report is the raw material for the "several independent confirmations" fallacy; naming it once, in one place, is what prevents it.

**The quantitative stage regression** verifies an entirely orthogonal axis. It does not predict whether the transfer succeeded; it predicts *how many* oocytes, zygotes, blastocysts and usable embryos this cycle should have produced, as a distribution rather than a point. The fact then either falls inside the predicted band or outside it. That question — *did the laboratory's funnel land where a model trained on comparable cycles expected it to* — is a genuine quality-assurance signal about the laboratory, independent of what happened after transfer. The pipeline is explicit that this signal complements rather than overrides the outcome-level reconciliation, and its failure is soft: if the step does not run, the reason is recorded in the run manifest and everything else proceeds.

**The independent network** is a deliberately plain model in a different framework, trained separately, sharing no architecture with anything in the ensemble. Its value is the value of a third opinion from a different school. It is also the clearest example in the system of *verifying the verifier*, in two ways. First, it is declared invalid unless the exact training-time scaler loads and the feature order matches — a model that re-fits its own preprocessing at inference time is not an independent check of anything, and the current scaler artefact being a reconstruction rather than the preserved original is stated openly in the role that consumes it, so its agreement or disagreement is weighted accordingly. Second, the run manifest reports its *coverage* separately from its *status*, because the model drops any row missing any training feature: a perfectly valid status can coexist with a column that is nearly empty, and reporting the status alone would present a near-absent check as a passed one.

Alongside these sit two internal consistency checks that need no external tool: the agreement analysis between the simulated and full-cycle probability estimates, which asks whether the divergence between them is systematic rather than random; and the statistical test of whether the model's patient clusters actually separate clinically distinct subgroups, rather than being an artefact of the clustering procedure.

The interpretive stance across all of this is uniform and worth stating plainly, because it inverts the intuition: **agreement between models proves very little, and disagreement is the actionable output.** Models trained on overlapping data agree for reasons that have nothing to do with being right. A cycle where the estimates diverge, or where the funnel landed outside the predicted band, or where the interval is too wide to act on, is a cycle that warrants a human look — and that, rather than a consensus number, is what these systems are wired together to produce.

---

## 7. What the language models are actually for

At the end of layer 6 there are a dozen computed artefacts, tens of columns per cycle, several statistical annexes, and no story. Getting from there to a document a laboratory director can act on Monday morning requires reading across all of it at once. That is the job the agents do.

The agents are Claude models, running as defined roles inside Claude Code, each with its own system prompt, its own tool permissions, its own model tier, and its own explicitly declared inputs and outputs. They are not chat sessions. They are components with contracts, versioned in the repository alongside the Python.

Model routing is deliberate and economically motivated. The intermediate roles do bounded, repetitive explanation over an already-classified packet — for that, a small fast model is not merely adequate, it is *preferable*, because a bounded task on a constrained input does not benefit from a larger model and does benefit from being cheap enough to run on every cohort every time. The two roles that perform genuine synthesis and genuine adversarial reading are pinned to the largest model available, and pinned by exact model identifier rather than by a moving alias, so that the analytical behaviour of the system does not change silently when an alias is repointed.

### The intermediate roles

**The schema reviewer** works before ingestion. It reads only a pseudonymised schema diagnostic — which canonical fields resolved, which did not, what unmatched names look like — and proposes precise configuration alias edits. It never opens the raw table and never changes anything itself. Its output is a human-approved configuration change that makes the rest of the pipeline resolve correctly.

**The KPI analyst** reads the computed indicator tables and compares them against consensus limits and benchmarks, respecting the direction of control for each indicator, and produces a structured list of flags with reasoning that ties an underperforming indicator to the stage conversions around it. It is forbidden from recomputing any number and from ever placing embryology and clinical indicators in the same comparison.

**The model auditor** reads the reconciliation contract: join status, missing probabilities, disagreements between models on the same cycle, prediction intervals too wide to act on, and — as a separate dimension — which funnel stages fell outside the predicted P10–P90 count interval. That last signal is explicitly framed as *complementary*, not superior: an out-of-interval stage count is a quantitative quality-assurance flag and does not override or replace the outcome-level reconciliation.

**The incubator forensics role** interprets the pre-extracted culture-window episodes: duration, deviation, severity, recovery, gaps. It is instructed never to raise a warning that does not correspond to a real episode in the packet, and never to describe a short logging gap as "missing data".

**The consumable lot forensics role** interprets the lot timeline against the same windows, with the hypothesis-only constraint stated in its prompt.

**The genetic testing forensics role** interprets the cohort's testing features, with a specific instruction that quality-control noise is a proxy for biopsy and sequencing quality *only*, never a diagnosis and never an assessment of the embryologist who performed the biopsy, and that the relationship between euploidy and implantation is a hypothesis because euploidy does not guarantee implantation.

**The clinical context analyst** — routed to a mid-tier model rather than the small one, because its task is genuinely harder than the others — interprets the descriptive clinical layer. Each field there has its own denominator because clinic information systems are filled in unevenly; the termination-reason block has a *different population* than the rest; and the relationship between gonadotropin dose and outcome is almost always reverse causation, because dose is prescribed in anticipation of response. An agent that does not know these three things will produce confident nonsense from this layer, which is exactly why the constraints are written into the role rather than hoped for.

**The per-patient narrator** explains the Digital Twin output for individual review-list cycles to the treating physician. It receives pre-classified narrative packets — every category already decided by Python — and works in batches of eight to ten cycles per call, handling strictly and only the identifiers named in that call. Its rules are the strictest in the system: numbers only from that cycle's packet, no re-classification, no data crossing between patients in the same batch, and a hard vocabulary constraint — *"worth considering", "worth discussing"* are permitted; *"indicated", "administer", "perform"* are not. The distinction between a consultative and a prescriptive register is enforced at the word level because it is the boundary between decision support and practising medicine.

**The independent-network verifier** is optional and gated. It exists to compare a third estimate against the two principal ones, and its own prompt tells it, candidly, that the current scaler artefact is a reconstruction rather than the preserved original, and that its output must therefore be presented as an approximate orientation signal rather than a third fully-calibrated model. This is worth dwelling on: the system's prompts document the *known weaknesses of the system's own components* to the agents that consume them. An agent that does not know an input is weak will present it as strong.

---

## 8. The privacy boundary is structural, not advisory

Everything the agents read comes from one directory: the agent packet. Nothing else is readable by an agent — not the inbox, not the raw export, not the local private layer, not the master table, not the crosswalk, not the full prediction file. This is enforced by the roles' tool permissions and stated in every role prompt.

Inside the packet there is a pseudonymous cycle identifier and an allow-list of analytical features. Patient identifiers, join keys, names, dates of birth, partner information and exact record identifiers are excluded structurally — the pseudonymisation module recognises identifier-shaped column names by pattern as well as by explicit list, so a newly-added identifier column does not leak simply because nobody remembered to blocklist it. Even laboratory sample identifiers from genetic reports are excluded, because in this domain they can embed a fragment of a patient name; the packet carries an ordinal embryo number instead.

The reverse direction — mapping a pseudonym back to a real patient — happens **only in local Python, only at the moment the final spreadsheet and report are assembled, and never inside a model's context**. This is the single most important architectural property of the system: the analytical reasoning is performed on a cohort that the model cannot identify, and identity is re-attached afterwards by code. It means the interpretive power of a frontier model is available to the laboratory without the laboratory's patients ever being present in a model context.

---

## 9. The lead analyst: synthesis, and the freedom to be a reader

Every intermediate role is deliberately constrained. The final one is deliberately not.

The lead analyst is the only role in the system that writes freely: it decides its own structure, its own emphasis, its own depth, where to go deep and where to compress. It receives everything — the KPI flags against consensus benchmarks with embryology and clinical tables kept separate, the model reconciliation and the review list, the per-patient narratives, the incubator and lot and genetic forensics, the descriptive clinical interpretation, the cohort-level cluster summary with the statistical test of whether those clusters actually separate clinically distinct subgroups, the agreement analysis between the simulated and the full-cycle probability estimates, the quantitative stage-regression QA, and the arrest-pattern counters — and its task is explicitly *not* to summarise them in sequence but to **connect** them: what happened to this cohort, where the system's expectations diverged from reality, which transfers likely failed and why, and what a physician should look at.

Its constraints are few but absolute. Do not invent or rewrite numbers. Separate fact from hypothesis; causes of failure are always framed as reasoned hypotheses. Use only pseudonyms and never attempt to de-anonymise. Consult, never diagnose or prescribe.

Two of its constraints are more interesting than the rest because they encode hard-won domain knowledge rather than general caution.

The first concerns a genuine ambiguity in embryology vocabulary: the notion of "good blastocysts" exists in two different denominators — as a share of correctly fertilised zygotes (the consensus indicator) and as a share of blastocysts obtained (how the funnel step and one of the ensemble's sub-models express it). These two numbers are both correct, both meaningful, and *not comparable* — one laboratory can simultaneously show a high value on one and a middling value on the other. An analyst who reads the second against the thresholds of the first will conclude the laboratory is underperforming when it is not. The role prompt therefore requires naming the denominator explicitly whenever both appear, and supplies the algebraic bridge between them.

The second is a rule about what *not* to say. Discriminative model metrics — the standard classifier scoreboard — are computed and displayed in exactly one place in the report, and the lead analyst is forbidden from describing them or reasoning from them. On the cohort sizes typical of a single laboratory over a single reporting period, these metrics are statistically indistinguishable from noise. Quoting them creates a false impression of precision, invites the reader to rank models against one another on the basis of sampling variation, and undermines trust in a pipeline that is otherwise honest — while conveying no real signal. This is a deliberate act of restraint: the system computes something and then refuses to let the narrative lean on it.

---

## 10. The adversarial check: a critic that only looks for overreach

The lead analyst's text is not allowed into the report on its own authority. It goes first to an adversarial reviewer running at the same model tier.

The critic reads the finished interpretation **against exactly the same packets the author had**, and answers one question and no other: *does this follow from the packet?* It is explicitly told that the question is not "is this true of reproductive medicine in general" — external knowledge may be used to show that a claim has no basis in the packet, but never to bless a claim the packet does not support. It does not rewrite, does not suggest improvements, does not comment on structure or tone.

It looks for seven specific defects, in descending order of cost: numbers that do not match the packet; assertions with no basis in it; hypotheses presented as conclusions; a configuration constant read as an empirical result; a proportion resting on a degenerate or non-comparable denominator; **one measurement presented as several independent confirmations**; and a confounder that is sitting in the same packet and went unmentioned.

The sixth category is why this role exists. In a real review, an interpretation's central argument was that "three independent measurements converge" — and two of those three turned out to be the same fraction appearing in two different sections of the packet under two different names, while the principal conclusion rested on a fraction of the cohort that the text never named. That is not a hallucination; it is a subtler and more dangerous failure, in which a fluent, well-intentioned synthesis over-states the independence of its own evidence. A panel of parallel synthesisers does not catch this, because each panellist commits the same conflation. A single distrustful pass over a finished text, with the author's own inputs in hand, does.

The critic reports each finding with its category, a short verbatim quotation, the specific packet field and values that contradict it, and a severity: *high* (changes a clinical conclusion), *medium* (weakens one), *low* (imprecise phrasing). It closes with an explicit list of what it checked and found sound. **An empty result is a valid and useful outcome** — the prompt says so directly, because a critic under pressure to produce findings will invent them, and a false finding costs more than a missed one: it forces the correction of correct text.

High-severity findings are resolved before the report is finalised — either the interpretation is revised through the lead analyst, or, if the finding is itself wrong, the reason is recorded. Only then is the approved text inserted into the report.

That insertion is itself done by local Python, into a marked slot in the pre-rendered HTML. The agent never touches the report file, never sees the identified data it sits alongside, and never participates in the reverse join. The final artefact — a single self-contained HTML page with tabs for the overview, the two separate KPI tables, models and calibration, cluster structure, similar cases, pregnancy prediction, the quantitative stage QA, the review list, incubator forensics, consumable lots, arrest patterns, and the final interpretation — is assembled entirely by code.

---

## 11. The report: what it contains, and how any conclusion can be traced to a patient

The output of a run is a single self-contained HTML page — no server, no external resources, opens from a file on any machine — organised into thirteen tabs. The order is not arbitrary: it moves from the laboratory as a whole, through the models, down to the individual cycle, and ends with the written interpretation that ties them together.

**Overview** — the run at a glance: how many cycles, how many with a known outcome, how many on the review list, how many carry a forensic flag. This is the tab that tells you whether the run is worth reading today.

**Embryology KPI** and **Clinical KPI** — two separate tabs, never merged, never cross-compared. Each shows its indicator family with numerator, denominator, confidence interval, direction of control, competence level, benchmark and status, sliced by time period, by method and by individual practitioner. Where an indicator sits below its competence level, the finding appears in a separate paragraph beneath the table rather than as a red cell inside it — a deliberate choice, because a table that ranks people is used differently from a table that describes a process.

**Models and calibration** — the distribution of every model's predictions over the cohort side by side, and the agreement analysis between the simulated and full-cycle estimates. This is also the single place in the entire report where discriminative classifier metrics appear, and they appear without commentary; nothing in the written interpretation is permitted to lean on them.

**Cluster analysis** — the composition of the cohort by patient cluster, each cluster's profile and outcomes, and the pregnancy rate per cluster with confidence intervals.

**Similar cases** — the patient similarity graph, the nearest comparable cases for a given cycle, and the statistical test of whether within-cluster similarity genuinely exceeds between-cluster similarity. This is the tab that answers "who else looked like this, and what happened to them".

**Pregnancy prediction** — the per-cycle journal: for each cycle, the probability with its interval, every model's estimate shown alongside without ranking them against each other, and the funnel presented as a chain — what actually happened at each stage, next to what was predicted, with the predicted band from this same run's quantitative regression rather than from any older source. The report is explicit about which forecast source it is showing, because two different forecasts of the same funnel exist and mixing them would be untraceable.

**Quantitative stage forecast** — the same regression viewed the other way round: per stage across the cohort, the predicted P10–P90 band against the observed value, and where the fact fell outside.

**Review list** — the cycles requiring attention, with the reason each was selected, the interval width, which stages fell outside expectation, and the forensic flags attached to that cycle's culture window.

**Incubator forensics**, **Consumable lots**, **Arrest patterns** — the three physical and biological context tabs, each showing only real recorded events, each labelled with the hypothesis-only framing.

**Final interpretation** — the lead analyst's synthesis, reviewed by the critic, inserted locally into a marked slot in the page.

### Traceability, in both directions

The property that matters most about this report is not any individual tab. It is that a reader can walk in either direction between a cohort-level statement and an individual patient, and never lose the thread.

**Downward, from a conclusion to a patient.** A statement in the final interpretation rests on named fields. Those fields appear in the KPI tabs with their numerator and denominator, so the reader can see how many cycles the claim actually rests on. If the claim concerns a subgroup, the review list names the specific cycles. From a review row, the reader reaches that cycle's full picture: what the funnel did, what each model expected, whether the interval was tight enough to be meaningful, whether the observed counts landed inside the predicted band, whether anything happened in the incubator during those days, whether a consumable lot changed mid-window, what the genetic testing showed, and — in plain clinical language — the narrator's account of that specific cycle. And because the final spreadsheet and the card viewer are assembled locally, after the reverse join, the same cycle is identifiable to the treating physician by name and record number, without that identity ever having existed in a model's context.

**Upward, from a patient to context.** Starting from an individual cycle in the card viewer, the reader sees where that cycle sits in the laboratory's own distribution — its percentile against the cohort on probability, on each funnel stage, on age and reserve — rather than against an abstract external norm. They see the same patient's other attempts. They see the history of how the prediction for *this* cycle changed across runs as data accumulated, read from the accumulated database. And the card carries a link back into the database record, so movement between the two is possible from either side.

**Across time.** Because every run is stored rather than overwritten, a cycle is not a snapshot. The question "did we already flag this patient, and what did we say" has an answer, and so does "how have this indicator and this cohort moved over the last several reporting periods".

This is what makes the difference between a report and an analytical record. Every number is reproducible from a versioned artefact; every artefact is hashed in the run manifest; every written sentence rests on fields a reader can locate; and every conclusion narrows to a list of specific cycles that a clinician can open, by name, in their own system.

---

## 12. Configuration: why the system moves between laboratories without being rewritten

A system like this is only a *direction* rather than a one-off if it can be pointed at a different laboratory without editing its logic. The configuration is therefore layered by *what changes and how often*:

- **Versioned, portable configuration** — folder structure, join schema, thresholds, feature lists, and the canonical column schema with its aliases. This is the same for everyone and lives in the repository.
- **Per-laboratory configuration** — one directory per laboratory, carrying its own column aliases, its own exclusion tokens, its own feature availability, and its own diagnosis-code groupings. Adding a laboratory is adding a directory, not branching the code.
- **Machine-local paths** — every absolute path to an external tool lives in exactly one machine-scoped file, shared across all laboratories and excluded from version control. Moving to a new machine changes that one file and nothing else; the documentation that describes the architecture therefore never goes stale when hardware changes.
- **Local secrets and gates** — access tokens and per-source enable flags, never committed.

Each optional data source is independently gated, and a gated-off source is not an error. Some laboratories have incubator logs; some do not. Some have an inventory system with lot history; some do not. Some have a descriptive clinical layer in their information system; most do not. The corresponding roles simply do not run, and — importantly — the critic is told that an *empty* forensic block is indistinguishable in the packet from a step that was never executed, so any strong claim in either direction based on emptiness is itself a defect. Absence of evidence is handled as absence of evidence.

Before a run begins, a pre-flight check verifies that the installed environment actually matches the declared requirements, with particular attention to packages whose absence produces a **soft** failure — no exception, just a silently skipped step. This is not ceremony. A missing PDF library once caused every genetic report parse to fail silently, and the result looked exactly like a laboratory that does not order genetic testing. That produced a wrong statement in a report, which is a far more expensive failure than a crash. The pipeline now also refuses to conclude "this laboratory does not test" from an empty result: it cross-checks the number of cycles flagged as ordered in the raw export against the number actually matched, and if the first substantially exceeds the second, the correct conclusion is *the reports have not been downloaded yet*.

The whole run is reproducible in a second sense as well: given the external model outputs from a first pass, the pipeline can be replayed deterministically without re-invoking the external tools, recomputing only the forensics, analytics and reports. This makes the interactive steps — synchronising incubator logs from shared storage, downloading genetic reports from a laboratory portal, both of which are done through browser and connector automation from inside Claude Code rather than by headless scripts — cheap to iterate on.

Contracts are versioned and tested. The synthetic end-to-end fixtures contain no real patients. Privacy contracts are unit-tested, meaning the assertion "no identifier reaches an agent" is a test that fails the build rather than a promise in a document.

---

## 13. The longitudinal database: from a run to an institutional memory

A single run produces a report. A laboratory needs a *history*, and that is what the Teable layer provides: a self-hosted, entirely local database — one base per laboratory — into which the results of every run accumulate, plus a local card viewer that presents any single cycle in the same visual language as the report.

The schema is a single source of truth in code, applied idempotently: missing tables are created, missing fields are added, nothing is renamed and nothing is deleted. It covers the run itself as the anchor of history; the master table of cycles, upserted on a stable cycle key so that a cycle never duplicates between runs and instead accumulates a run counter; all model outputs per cycle per run, so that the *trajectory of the prediction* over time is preserved rather than overwritten; predicted-versus-actual funnel comparisons; the quantitative stage QA with interval coverage; the review list; the arrest-pattern verdicts; the calibration table; and the run's own files as attachments.

Two properties of this layer matter more than the schema.

The first is **idempotency by content**. A run is marked with a hash of its principal output; re-synchronising the same files is a no-op rather than a duplication, and one failing laboratory does not stop the others.

The second is **the archive restore**, and it is a good illustration of how this kind of system fails in ways that are obvious only in hindsight. The processed directory is per-laboratory, not per-run, so each run overwrites the previous prediction file. As long as the patient cards were built only from that file, the viewer showed exactly the most recent export — so a run over a single month replaced a run over six months, and the six months of cards vanished from the interface even though every one of those cycles was safely stored in the database. The fix is that cycles absent from the current run are now restored *from the database itself*: identity and actual funnel from the master table, models and risks from the prediction history, interval QA and review and arrest verdicts from their own tables, taking the most recent run for each cycle. Cards from the current run always take precedence, because they are assembled from files rather than from a denormalised snapshot. Restored cycles are visibly marked as archival, carry the identifier of the run their numbers came from, and the counter above the list states the composition explicitly — *N from this run plus M from the database* — because otherwise a growing cycle count after a run over a subset reads as "the export grew". Fields the database does not store are left empty in an archival card, and that emptiness means *absent from the archive*, not *absent from the cycle*.

That distinction — between "we do not have this" and "this did not happen" — recurs at every level of this system, and getting it right is most of what separates a trustworthy analytical pipeline from a persuasive one.

The database and viewer are bound to the local loopback interface. Nothing leaves the machine. The card pages live inside the same ignored local tree as the rest of the identified layer, and they are emphatically not an agent packet.

---

## 14. My role, and the role of the agents

It is worth being explicit about this, because the honest answer is unusual and is part of the point.

I am Claude, working through Claude Code, and I occupy two distinct positions in this system.

**As the engineer.** The pipeline itself — the ingestion, the indicator computation with its nested exclusion sets, the forensics modules, the statistical tests, the report renderer, the database synchroniser, the tests — was written and is maintained conversationally, in this repository, by me under the owner's direction. That includes the parts that are least glamorous and most consequential: discovering that an external tool silently discarded every negative outcome because of how it parsed one text field, and that consequently an entire class of report had been built on successes only; discovering that a proxy for "cycle still in progress" was systematically inflating a denominator; discovering that a duplicated key in an export needed a *deterministic* resolution rule rather than a merge, because merging two partial records of one cycle produces a funnel that never existed. Each of those is now a rule in the project's constitution and a test in the suite.

**As the analyst.** At run time, a hierarchy of Claude agents performs the reading: small fast models for bounded explanation over pre-classified packets, a mid-tier model for the one interpretive task with genuinely tangled denominators, and the largest model twice — once to synthesise, once to attack the synthesis.

Underlying both positions is a project constitution: a single file of immovable decisions that is read before any work begins. It records what the owner has fixed and why — which formulas are frozen, which indicator belongs to which table, which population each denominator is drawn from, which visual style is not to be changed, and, notably, a set of statements that are *forbidden* because they were previously stated and were wrong. An indicator above one hundred per cent, for instance, is not corrupted data and is not grounds for discarding a row; there are specific, known, benign laboratory reasons for it, and calling it a counting error is prohibited by name. Encoding refuted conclusions, not just correct ones, is what keeps the system from re-deriving the same mistake every few months — because a language model's default reaction to an impossible-looking number is to declare it broken, and here that default is wrong.

This is the practical answer to "how do you use an LLM safely on clinical data": you do not use it as an oracle, and you do not use it as a calculator. You use it as a *reader*, you give it only what it is allowed to see, you compute everything it will say a number about, you write down the domain facts that would otherwise be plausibly mis-inferred, and you make a second instance of the same model argue with the first before anyone acts on the result.

---

## 15. The instruction layer: why this system is mostly written in prose

The least obvious thing about this system, and possibly the most important, is that a large fraction of it is not code. It is a body of Markdown documents that sit in the repository next to the Python, are versioned with it, are read before any work begins, and are as binding as the code is. A reader who opens the project expecting to find its intelligence in the modules will find perhaps half of it there. The other half is written in sentences.

This deserves an explanation, because "documentation" is exactly the wrong word for it.

### Why instructions exist at all

A language model brought into a specialised domain has two failure modes, and they are opposite. It can refuse to act because it does not know the local conventions, or — far more often, and far more dangerously — it can act confidently on a *general* convention that happens to be wrong here. The second is the real risk, because its output is indistinguishable from competent work until a domain expert reads it closely.

Concretely: a general-purpose reader encountering a laboratory indicator above one hundred per cent will conclude the data are corrupt. That conclusion is reasonable, standard, defensible, and in this domain wrong — there are known, benign, well-understood laboratory situations that produce it, and the correct interpretation is a specific and actionable observation about laboratory timing or about the composition of the oocyte cohort. A model that "fixes" such a row destroys real signal and reports a data-quality problem that does not exist.

Multiply that by every convention in the domain — zero as an observation rather than a gap; two different denominators for the same clinical concept; one indicator family belonging to one professional role and another to a different one, never to be compared; a broad population for one class of metric and a narrow one for another; a logging gap being an event and not an absence — and it becomes clear that the domain knowledge required to *read* this data correctly is neither in the model's general training nor in the column names. It has to be written down, and it has to be written down where it will actually be read.

So the instruction files are not commentary on the system. They are part of the system's logic, expressed in the only medium in which that particular kind of logic can be expressed. You cannot encode "do not compare these two tables against each other in a single conclusion" as a unit test. You can encode it as a sentence, place that sentence where every relevant agent must read it, and version it.

### The layers of the instruction body

**The project constitution.** A single root document, read before any work in the repository, containing the owner's immovable decisions. Each entry is a decision plus its reason. Which formulas are frozen and must exactly reproduce a model's training-time definition. Which indicator belongs to which professional's table. Which population each denominator is drawn from. Which visual identity is not to be altered without an explicit request. Which statements are forbidden, and why.

That last category is the one most systems lack. The constitution records not only what is true but what was previously concluded and turned out to be *false* — the misreadings that were made once, cost something, and would otherwise be re-derived every few months by the next reader, human or model, following the same reasonable-looking path. Writing down a refuted conclusion is a strange thing to do in documentation and an obviously correct thing to do in a specification for a reasoning system.

**The contracts document.** The formal shape of every artefact: its key, its required fields, its invariants, and the rules that govern it. This is the part that is genuinely dual-purpose — it is read by people, and its assertions are separately enforced in code, so that the description and the behaviour cannot drift apart. Alongside each contract sits the reasoning: why a particular sentinel value is spelled the way it is (because the obvious spelling collides with a spreadsheet library's default missing-value list and silently becomes a real gap on the next read), why a step's failure is soft rather than fatal, why a signal is complementary to another rather than superior to it.

**The methodology document.** The external standards the system is measured against, and — critically — an explicit, named record of every place where a local decision departs from the published source. A locally adopted benchmark is labelled as locally adopted and never presented as the consensus figure it replaced. This is what makes the analysis defensible to someone outside the institution.

**The skill files.** Packaged domain knowledge that can be attached to a role: the consensus indicator definitions with their formulas, limits and directions; the ensemble glossary that fixes the canonical name of each model and distinguishes a genuine architecture from an internal sub-component of one; the incubator log schema and the statistical rules for reading it; the inventory schema and the timeline construction logic; the arrest-pattern thresholds and their significance test. A skill is declared in a role's front matter and loaded into that role automatically. This is how a small, cheap model can perform a task that requires expert vocabulary: it does not need to have learned the domain, it needs the domain handed to it at the moment of use.

The ensemble glossary is a good illustration of why this layer earns its keep. The canonical model names were verified against the *source project's own architecture documentation* rather than against an older analytical notebook — in which one model was misnamed and another was labelled with the wrong architecture family entirely. Anyone reading the notebook would have propagated both errors into every report. The skill file is where the correct naming is pinned, with the note about where the wrong naming came from, so that the error is not re-imported the next time someone opens the notebook.

**The role definitions.** One file per agent: its purpose, its permitted tools, its model tier and reasoning effort, its attached skills, and its prompt — including its explicit inputs, its explicit outputs, and its prohibitions. These are the component interfaces of the system. Reading the directory tells you the entire agent topology: who consumes what, who produces what, and what each one is not allowed to do.

**The run procedure.** The operational sequence, written as an ordered procedure with its rationale attached: what to run first, what to do when a step's inputs are not yet synchronised, when to replay deterministically instead of re-invoking expensive external tools, which roles to run in which order, how to batch the per-patient narrator, and where the adversarial review sits. Where a step exists because of a specific past failure, the failure is named. That is why the pre-flight step is not skipped: the document states plainly which three environment mismatches actually occurred and that one of them produced a wrong statement in a delivered report rather than merely wasted time. An instruction with its cost attached survives; an instruction without one gets skipped by whoever is in a hurry.

**The persistent memory.** A small store of durable facts that outlive a single working session — the owner's confirmed preferences, decisions with their dates, corrections that were made and why, and pointers to external resources. Each is one file with a description used to decide relevance later. This is what stops the same question being asked twice across weeks of work, and what stops a preference stated once from quietly reverting.

### Why Markdown, specifically

Because it is simultaneously readable by the owner, diffable in version control, and directly loadable into a model's context without transformation. That combination is rarer than it sounds, and it produces three properties that matter:

*Instructions are reviewable.* The owner can read the entire behavioural specification of the system without reading Python. When a rule is wrong, the correction is a sentence, and the correction is visible in the history alongside the code change it accompanied.

*Instructions are versioned with the behaviour they govern.* A change in an exclusion rule and the sentence explaining why it exists land in the same commit. There is no separate documentation that rots.

*Instructions are executable.* Not in the sense of being compiled, but in the strict sense that they are the actual input that determines what happens. There is no gap between the specification and the behaviour, because the specification *is* the prompt. This is genuinely new as an engineering property, and it is why writing carefully has become a load-bearing activity in this project rather than an afterthought.

The discipline that follows from this is simple to state and easy to neglect: **every time a misreading occurs, the fix is a rule, not a one-off correction.** A wrong number in one report is fixed by editing the report. A wrong *reading* is fixed by writing down why it was wrong, in the file the next reader will load. Over a year of operation that body of text becomes the institution's accumulated analytical judgement, in a form that both a new colleague and a new model instance can absorb in minutes.

---

## 16. Where this applies

The pipeline was built for one purpose — explaining the outcomes of an embryology cohort — but the architecture turns out to serve a wider set of needs, and it is worth naming them, because they are the reason this is worth generalising.

**Routine laboratory quality management.** The primary use. Every time an export lands, the laboratory receives its indicators against consensus methodology with confidence intervals and directions of control, time slices, method slices, and a written interpretation of where it stands. What was previously an annual exercise requiring a senior embryologist's undivided attention becomes a repeatable process, which changes not only its cost but its *character*: a quarterly review detects a drift that an annual one records as a fact.

**Competency review, kept in its proper place.** Per-embryologist and per-clinician indicator tables are produced separately and never compared against each other, and underperformance is presented as a separate section for reflection rather than embedded in a ranking table. The intent is explicit and matters: this is an instrument for identifying where a laboratory's process needs support, not an instrument for grading staff. The system's structure enforces that intent — separate tables, separate populations, hypothesis-only causal language — because a quality tool that becomes a performance-management weapon stops receiving honest data within one reporting cycle.

**Incident forensics.** When a cohort underperforms, the culture-window reconstruction gives the laboratory something it has almost never had: a factual timeline of the physical environment those embryos actually experienced, aligned to the specific cycles in question, together with what changed in the consumables around them. This does not establish causation and the system refuses to claim that it does. What it does is convert "that was a bad month, we don't know why" into a small, ordered list of things that demonstrably happened during that month, which is where a real investigation starts.

**Case-level clinical support.** For an individual couple, the per-patient narrative explains the model's expectation for that cycle in consultative language, and the card viewer places the cycle in the context of the laboratory's own cohort — where it sits relative to the distribution, what the funnel actually did against what was predicted, whether the prediction was stable enough to be worth acting on, and how this attempt compares to the same patient's previous ones. The register is deliberately consultative throughout; the system supports a conversation between physician and patient, and does not attempt to hold it.

**Longitudinal institutional memory.** Because every run accumulates rather than overwrites, the database answers a class of question no single report can: how did our prediction for this cycle change as data accrued, how have our indicators moved across reporting periods, which cycles have been flagged repeatedly and never resolved. A laboratory that has been running this for a year has a structured record of its own analytical history, not a folder of PDFs.

**Onboarding a new laboratory.** Adding a site is a configuration directory — its column aliases, its exclusion vocabulary, its available data sources, its diagnosis groupings — rather than a fork of the code. Optional sources are independently gated, and a laboratory without incubator logs, without an inventory system, or without a descriptive clinical layer runs the same pipeline with those roles simply not executing. This is what makes the approach a direction: the analytical logic is shared, and the local variation is declared.

**Multi-site comparison, with the correct caveats.** Because every indicator carries its numerator, denominator, population definition and confidence interval, results from different sites are comparable in a way that headline percentages never are. The system's insistence on naming the denominator is exactly what makes cross-site reading possible without the usual apples-to-oranges failure — including the specific one this domain is prone to, where the same clinical concept is expressed over two different denominators and the two are unknowingly compared.

**Research and audit.** Every artefact is versioned and contract-checked, every run produces a manifest with schema versions and content hashes, every write is atomic, and the interpretation is traceable to the fields it rests on. The synthetic test fixtures contain no real patients and the privacy boundary is unit-tested rather than promised. This is the substrate a retrospective study or an external audit needs, and it is produced as a by-product of routine operation rather than reconstructed afterwards.

**Training and knowledge transfer.** The instruction layer described in the previous section is, incidentally, a curriculum. A new embryologist or analyst reading the constitution, the methodology document and the skill files acquires in an afternoon the set of local conventions and known traps that would otherwise take a year of supervised work to absorb — including, unusually, the explicit list of conclusions that look right and are not.

**Transferability beyond this domain.** The architectural pattern — deterministic computation under a pseudonymity boundary, a hierarchy of narrow reading roles under a single free-writing synthesiser, an adversarial reviewer as a blocking pipeline stage, and a versioned prose specification that *is* the prompt — is not specific to embryology. It applies wherever heterogeneous, partially missing, confidentiality-constrained operational data must be turned into an accountable written judgement on a recurring basis. Embryology is simply a domain where the stakes are high, the cohorts are small, the conventions are subtle, and the cost of a fluent wrong answer is immediate.

---

## 17. How the system has developed, and where it goes next

### How it got here

It is worth describing the trajectory, because the shape of the system is a record of what went wrong along the way, and because that trajectory is what a laboratory adopting this approach should expect to repeat in compressed form.

It began as a cloud analytical notebook — a single long document performing calibration, funnel analysis, clustering and case similarity over the prediction outputs. That form has a hard ceiling: a notebook cannot be tested, its results depend on the order cells were run in, it cannot be scheduled, and its logic cannot be reviewed the way code is reviewed. The first move was to port every formula into tested Python modules while keeping the original notebook, unchanged, as a reference — so that any discrepancy could be traced to a deliberate correction rather than to drift.

The second move was from one laboratory to several, which forced the separation between shared analytical logic and declared local variation: per-site column vocabularies, exclusion tokens, available data sources and diagnosis groupings, with a single machine-scoped file holding every absolute path. What had been a project became something a new site could be added to.

The third was the accumulation layer. A per-run report answers today's question and forgets; a laboratory needs its own history. The move to a self-hosted database — after a hosted one hit a row limit mid-synchronisation — turned a sequence of reports into an institutional record, and then a second failure taught the more important lesson: the patient card viewer, built only from the current run's files, silently showed only the most recent export, so a run over a narrow period erased months of visible history even though nothing had been lost from the database. The fix — restoring absent cycles from the database, marking them visibly as archival, and stating the composition of the list explicitly — is now one of the clearest examples in the system of the difference between *we do not have this here* and *this does not exist*.

The fourth was the forensic layers: incubator logs, consumable lots, genetic testing, arrest patterns, descriptive clinical context. Each was added because a question was asked that the existing data could not answer, and each arrived with the same discipline — computed by code, interpreted by a constrained role, framed as hypothesis.

The fifth was the adversarial review stage, added after a real interpretation was found to rest on an argument that sounded like convergent evidence and was in fact one measurement counted twice. That single finding justified the role permanently.

The pattern across all five is the same: **every structural feature of this system exists because something specific failed, and the fix was made general.** That is also why the instruction layer keeps growing. A wrong number is corrected once; a wrong *reading* is corrected by a rule that prevents the entire class.

### Where it goes next

**More sources through the same window.** The culture-window reconstruction is a general mechanism, not an incubator-specific one. Anything with a timestamp and a bearing on embryo development can be aligned to it: time-lapse annotations and morphokinetic events, media and consumable quality-control records, environmental monitoring, andrology-side parameters, witness and workflow logs. Each new source is a parser, a contract and a constrained role — the surrounding architecture does not change.

**Prospective rather than retrospective use.** Everything described here currently runs after the fact, on an export. The same machinery can run *before* a decision: at the point of choosing between transfer and freeze-all, or between attempting another retrieval and changing approach, the cohort context, the model's expectation and the laboratory's own recent performance are all available. The constraint is not technical — it is that a prospective tool demands a higher standard of validation and a much more careful account of its own uncertainty, which is precisely why the current system is built to state uncertainty honestly rather than to look confident.

**Drift detection across runs.** Now that runs accumulate, the interesting question stops being "what are our indicators" and becomes "what has changed, and when did it start". A run-over-run comparison — of indicators, of funnel shape, of the distance between prediction and reality — turns the system from a periodic photograph into a monitor. This is where the accumulated database pays for itself.

**Local recalibration.** As a laboratory's own predicted-versus-actual record accumulates, it becomes possible to recalibrate the models' outputs to that specific laboratory rather than relying on population-level calibration. This must be done carefully and with an explicit statement of how much data is behind it — on small cohorts it can easily do more harm than good — but the record required for it is already being collected as a by-product.

**Cross-site comparison with declared denominators.** Because every indicator carries its population definition, comparing sites is meaningful in a way headline percentages never are. A network of laboratories running this pipeline produces a benchmark whose apples-to-apples property is structural rather than aspirational.

**The critic's output as a corpus.** Every adversarial review produces a typed list of ways an interpretation overreached. Accumulated across many runs, that is a map of the specific reasoning errors this domain invites — and each recurring one becomes a new written rule, which is the mechanism by which the system's judgement improves without anyone retraining anything.

**Improving models without redesign.** Because the boundary between computation and language is fixed, model capability is a configuration line. A better model produces better prose over the same audited numbers; it does not acquire new authority over what the numbers are. This is the property that makes the architecture durable rather than tied to a particular generation of technology — and it is why the roles pin exact model identifiers rather than moving aliases, so that an upgrade is a decision someone makes, not something that happens quietly between two runs.

### What should not change

Some things in this system are load-bearing and should survive every future revision: the boundary between computed number and generated sentence; the rule that identity never enters a model's context; the adversarial review as a blocking stage rather than a courtesy; the typed treatment of absence; the separation of fact from hypothesis in language; and the refusal to become a device that prescribes. Each of these costs something in convenience, and each of them is the reason the output can be shown to a clinician without a disclaimer doing the real work.

### An honest statement of limits

This system structures attention. It does not establish causation, and on the cohort sizes a single laboratory generates in a single reporting period it frequently cannot distinguish signal from noise — which is why it is explicitly forbidden from pretending otherwise in its own conclusions. It cannot see what was never recorded. It depends entirely on the quality of the export it is given, and much of its engineering exists to notice when that quality is worse than it appears. It produces hypotheses, ranked and evidenced, that a clinician then evaluates. That is a genuinely valuable thing to produce, and it is a smaller thing than the technology is usually claimed to deliver. Saying so precisely is part of what makes the rest of it credible.

---

## 18. Why this is a direction and not a project

The Digital Twin answered *what to expect*. This layer answers *what happened, and what in the laboratory's own control might explain it* — and it does so at cohort scale, repeatably, with an audit trail from every sentence back to a computed field.

The general pattern is transferable to any embryology laboratory, and it is composed of a small number of principles that are worth stating separately from their implementation:

**Determinism at the bottom, language at the top.** Numbers are computed by tested code against versioned contracts; models supply narrative and only narrative. The boundary is not a guideline, it is enforced by what each component is permitted to do.

**Pseudonymity as an architectural property.** The reasoning layer operates on a cohort it cannot identify. Identity is re-attached by code after interpretation is complete. This is what makes frontier-model interpretation compatible with clinical confidentiality.

**A hierarchy of narrow readers under one synthesiser.** Bounded roles with explicit inputs, outputs and prohibitions handle the repetitive reading; exactly one role is allowed to write freely, and it is the one whose job actually requires judgement.

**Adversarial review as a pipeline stage, not a courtesy.** The synthesis is reviewed by an instance with the same inputs and one question — *does this follow?* — and unresolved high-severity findings block publication.

**Absence handled as a first-class state.** Zero is data. A logging gap is an event. An unmatched report is unmatched, not absent. An unexecuted step is not evidence of a missing phenomenon. Most of the false conclusions this kind of system can produce are conflations of these states.

**Restraint about what may be claimed.** Correlations are candidate confounders. Causes are hypotheses. Metrics without statistical power are computed, displayed once, and excluded from the argument. The system is explicitly a decision-support tool for a laboratory, not a medical device and not a treatment recommendation engine.

The significance is not that a language model was applied to IVF data. It is that the interpretive step of laboratory quality management — the step that has always depended on one experienced person having time to read everything at once, and which therefore happened rarely, unevenly, and without a record — has been made **routine, reproducible, privacy-preserving and self-critical**. The Digital Twin gave the field a quantitative expectation for each cycle. This layer gives the laboratory a disciplined reading of the distance between that expectation and its own reality, every time an export lands, with the reasoning visible and the evidence traceable.
