# Every reviewed error case

October 6, 2026. These are label-aware model-based diagnostic notes, not replacement labels or independent human review. Start with [the simple review report](ERROR_REVIEW.md). All 37 questions with at least one mismatch are covered. Candidate content was treated as untrusted data; most external facts were not checked.

In each pair, the first choice is from original order and the second is the swapped choice mapped back to the original answer identity. `tie` means the existing `tie_preference` label. Raw displayed choices remain in the saved prediction table.

## helpsteer2-preference-8680 — Amplitude definition

Publisher label: **B**. Baseline **A/B**; 20 updates **tie/B**; 200 updates **A/B**.

Both interpret the ambiguous question as a physics term. B offers a wave/signal definition and more context; A gives a shorter definition with units needing checking. The model selects the displayed first answer in both orders. The definition and unit claims were not independently verified in this review.

Review tags: external_facts_unchecked, coverage_difference. Flags: order disagreement. Released individual preference strengths: [2, 1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8683 — Coaching book contents

Publisher label: **A**. Baseline **A/tie**; 20 updates **tie/tie**; 200 updates **tie/tie**.

Both provide English book outlines covering coaching relationships, development, and ethics. A names more chapters and appendices; B groups topics more compactly. A strong objective advantage is not established by the prompt alone. All later decisions are ties, so this is a stable disagreement with the publisher label, not a swap failure.

Review tags: preference_ambiguity. Flags: stable mismatch in both orders. Released individual preference strengths: [-1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8693 — Answer the accusation, not the old R question

Publisher label: **A**. Baseline **B/B**; 20 updates **B/B**; 200 updates **B/B**.

The latest user turn asks whether the prior response was an honest mistake or gaslighting. A directly addresses that accusation; B briefly apologizes and returns to a long, repetitive package overview. The model consistently prefers B in all three versions. Package capabilities were not verified, and no code was executed.

Review tags: latest_turn_relevance, external_facts_unchecked. Flags: stable mismatch in both orders. Released individual preference strengths: [-3, 1, -3, -3, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8697 — Lighthearted fashion jokes

Publisher label: **tie_preference**. Baseline **B/B**; 20 updates **tie/B**; 200 updates **B/B**.

Both provide suggested jokes and warn against belittling the other person. A lists many generic compliments; B organizes a smaller set by approach. Humor and tone permit subjective preferences. The original tie match after 20 updates becomes a stable preference for B after 200.

Review tags: preference_ambiguity, overlapping_content. Flags: original-order regression from 20 updates. Released individual preference strengths: [1, 1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8702 — Imagined Caesar/Alexander interview

Publisher label: **tie_preference**. Baseline **B/B**; 20 updates **A/B**; 200 updates **B/B**.

Both satisfy the fictional interview format and discuss public opinion. A emphasizes leadership principles; B adds historical-sounding symbols and campaigns. Those historical details were not checked. The stable B preference is a label mismatch, not established proof that B is factually inferior.

Review tags: preference_ambiguity, external_facts_unchecked. Flags: stable mismatch in both orders. Released individual preference strengths: [-1, 1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8707 — Praline scanning blog invents claims

Publisher label: **A**. Baseline **B/B**; 20 updates **B/B**; 200 updates **B/B**.

A invents nutritional-information and healthier-choice benefits absent from the supplied demo description. B invents a training set of over 1,000 images and over 95% accuracy. Both go beyond the supplied facts. The model consistently prefers B although the publisher label is A. Preference scoring cannot certify either answer as supported; preserve the label and request future independent adjudication.

Review tags: unsupported_additions, human_adjudication_needed. Flags: stable mismatch in both orders. Released individual preference strengths: [-1, -2, -2].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8710 — Life-insurance categories

Publisher label: **tie_preference**. Baseline **A/B**; 20 updates **A/tie**; 200 updates **A/tie**.

Both list overlapping insurance categories but use different fifth categories and different explanations. The original A preference changes to a mapped tie when swapped. This review compares coverage and output behavior only; insurance claims were not verified and these answers are not being recommended as financial advice.

Review tags: overlapping_content, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [-1, 1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8747 — Spell-check a website by URL

Publisher label: **A**. Baseline **A/B**; 20 updates **A/tie**; 200 updates **A/B**.

The request specifically asks for a tool that accepts a website URL. Both mostly list writing tools or extensions without demonstrating that workflow; B duplicates a product family and adds extension claims. Availability and URL-scanning support were not checked. All versions show order dependence, so selecting the publisher label does not verify the tools.

Review tags: constraint_mismatch, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [-2, -2, -3].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8753 — Satirical orange-box skit

Publisher label: **A**. Baseline **B/B**; 20 updates **B/B**; 200 updates **B/B**.

A acts out tourists trying to pay with a credit card and adds a pretend-validation joke. B uses a longer dialogue with repeated sarcasm labels and mentions the credit-card behavior. Both respond to the requested rewrite, but humor quality is subjective. The stable B choice persists across all versions despite label A.

Review tags: preference_ambiguity, latest_turn_relevance. Flags: stable mismatch in both orders. Released individual preference strengths: [1, 3, -3, -1, -2].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8761 — Architect prioritizes risky work

Publisher label: **tie_preference**. Baseline **B/B**; 20 updates **tie/tie**; 200 updates **tie/B**.

Both explain prioritizing uncertain work and deferring a low-risk implementation choice. B adds concrete examples; A includes more speculation about why the architect did not answer. Their core explanations overlap. The 20-update tie in both orders becomes tie/B after 200 updates.

Review tags: overlapping_content, preference_ambiguity. Flags: swapped-order regression from 20 updates, order disagreement. Released individual preference strengths: [1, 1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8810 — PWA characteristics

Publisher label: **tie_preference**. Baseline **A/B**; 20 updates **tie/B**; 200 updates **B/B**.

Both list web technologies, broad device support, performance, installation, and security. B adds maintenance and distribution claims. The released reviewers disagree over whether that extra coverage helps. Technical claims were not independently checked; the strict equality wording is narrower than the publisher's about-the-same tie definition.

Review tags: overlapping_content, external_facts_unchecked. Flags: original-order regression from 20 updates. Released individual preference strengths: [1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8844 — Redis persistence

Publisher label: **A**. Baseline **A/B**; 20 updates **A/tie**; 200 updates **A/B**.

B claims Redis does not persist data to disk. Official Redis documentation explicitly describes RDB and AOF persistence, contradicting that claim. This does not certify every claim in A, including its platform and memory descriptions. The baseline and 200-update versions choose whichever answer is first, selecting B when it moves to the first position; the 20-update version gives A/tie after mapping back.

Review tags: source_checked_error, human_adjudication_needed. Flags: order disagreement. Released individual preference strengths: [-1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8849 — Generic CEO role prompt

Publisher label: **tie_preference**. Baseline **A/B**; 20 updates **tie/B**; 200 updates **B/B**.

The prompt supplies a role and company name without a concrete requested deliverable. Both invent plausible CEO priorities and company assertions; one uses paragraphs and the other a numbered list. The company claims were not checked. The original tie match at 20 updates becomes a stable B choice.

Review tags: preference_ambiguity, unsupported_additions. Flags: original-order regression from 20 updates. Released individual preference strengths: [-1, 1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8854 — Acknowledge incoming messages

Publisher label: **tie_preference**. Baseline **B/B**; 20 updates **tie/tie**; 200 updates **tie/B**.

Both acknowledge the request and invite the user to continue sending information. B more explicitly promises acknowledgment of each message; neither performs legal analysis. Their memory promises are not a verified capability. The final pair changes from a tie in original order to B when B is first.

Review tags: overlapping_content. Flags: swapped-order regression from 20 updates, order disagreement. Released individual preference strengths: [1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8864 — Reflective writing examples

Publisher label: **B**. Baseline **A/B**; 20 updates **tie/tie**; 200 updates **tie/B**.

Both describe reflective writing. A organizes examples around cases and incidents; B gives diaries, essays, portfolios, and group discussion. The publisher prefers B, while the model alternates tie/B depending on position. Professional-practice claims were not independently verified.

Review tags: coverage_difference, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [1, 1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8867 — Fintech business-plan request

Publisher label: **tie_preference**. Baseline **B/B**; 20 updates **A/B**; 200 updates **B/B**.

Both mostly tell the user what a business plan should contain rather than drafting the requested restaurant credit-line business with the named distributor. B has more headings, but neither develops the specific transaction flow. Equal preference need not mean either answer is adequate. The final model stably chooses B.

Review tags: constraint_mismatch, overlapping_content. Flags: stable mismatch in both orders. Released individual preference strengths: [1, -1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8875 — GPA category

Publisher label: **A**. Baseline **A/B**; 20 updates **tie/B**; 200 updates **A/B**.

Both give the same requested category, math calculators. A additionally suggests home or personal-finance categories without a clear connection to the categorization task. That makes the publisher's A preference worth independent review rather than automatic relabeling. The model chooses the displayed first answer in either order.

Review tags: overlapping_content, human_adjudication_needed. Flags: order disagreement. Released individual preference strengths: [-1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8881 — Moving-business invoice

Publisher label: **A**. Baseline **A/B**; 20 updates **A/tie**; 200 updates **A/tie**.

A includes invoice identifiers and service lines; B includes move-specific fields such as dates, addresses, movers, and hours. Both add policies the user did not supply. Legal enforceability and suitability of those policies were not reviewed. The final A/tie pair is an order-dependent preference, not an invoice compliance check.

Review tags: preference_ambiguity, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [-2, -1, -2].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8884 — Vue introduction

Publisher label: **B**. Baseline **tie/tie**; 20 updates **tie/tie**; 200 updates **tie/tie**.

Both explain Vue in terms of user interfaces and reactive data, with differing feature coverage and promotional claims. All versions give ties in both orders despite label B. API and performance claims were not independently checked; this is a stable preference mismatch.

Review tags: overlapping_content, external_facts_unchecked. Flags: stable mismatch in both orders. Released individual preference strengths: [1, 1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8888 — Backend-driven frontend

Publisher label: **tie_preference**. Baseline **A/tie**; 20 updates **tie/tie**; 200 updates **A/tie**.

Both describe a frontend receiving backend data and discuss separation of responsibilities. B adds scalability and security claims. The prompt's architectural term may call for a more specific explanation; this review does not certify either definition. The original tie becomes A while the swapped decision remains a tie.

Review tags: overlapping_content, external_facts_unchecked. Flags: original-order regression from 20 updates, order disagreement. Released individual preference strengths: [1, -1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8890 — Focused input appearance

Publisher label: **tie_preference**. Baseline **A/tie**; 20 updates **A/tie**; 200 updates **A/tie**.

Both answer yes and discuss visible focus. B also connects focus with successful task completion and auditory cues, claims needing a separate accessibility check. The model produces the same A/tie pattern in all three versions. This is not evidence that fine-tuning newly introduced the problem.

Review tags: overlapping_content, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8897 — Interview thank-you letter

Publisher label: **tie_preference**. Baseline **A/B**; 20 updates **A/tie**; 200 updates **A/B**.

Both are usable letter templates expressing thanks and interest. B includes more personalization placeholders, while A is shorter. The prompt does not demand a particular length or those extra details. The expanded model picks whichever letter is displayed first.

Review tags: preference_ambiguity, overlapping_content. Flags: swapped-order regression from 20 updates, order disagreement. Released individual preference strengths: [1, -1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8910 — Teams external registration error

Publisher label: **B**. Baseline **A/B**; 20 updates **A/B**; 200 updates **A/B**.

Both propose troubleshooting, while B names more specific UI options and browser checks. Whether those settings exist and resolve the historical issue was not verified. Both-order first-position choices occur in every model version, so the publisher preference alone is not a validated repair procedure.

Review tags: external_facts_unchecked, coverage_difference. Flags: order disagreement. Released individual preference strengths: [1, 1, 2].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8915 — Rebranding a shared design system

Publisher label: **tie_preference**. Baseline **A/B**; 20 updates **A/tie**; 200 updates **A/tie**.

Both cover cross-application consistency, component changes, testing, and communication. A stresses dependencies; B explicitly adds accessibility testing. Neither provides a measured cost estimate. The final A/tie pattern shows a preference boundary changing with position despite substantial content overlap.

Review tags: overlapping_content. Flags: order disagreement. Released individual preference strengths: [-1, -1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8928 — Compare two strings

Publisher label: **tie_preference**. Baseline **A/B**; 20 updates **tie/tie**; 200 updates **A/tie**.

Both offer several string-comparison approaches and share multiple methods. Their lists differ, and B asks for clarification before answering. No implementation is supplied or executed. The original tie at 20 updates becomes A at 200, while the swapped decision remains a tie.

Review tags: overlapping_content, external_facts_unchecked. Flags: original-order regression from 20 updates, order disagreement. Released individual preference strengths: [1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8945 — Browser-plugin report

Publisher label: **tie_preference**. Baseline **B/B**; 20 updates **B/B**; 200 updates **B/B**.

Both follow introduction/body/conclusion and cover the requested three capabilities. A adds unrequested features; B adds blanket privacy and compliance promises without supporting design details. The model stably prefers B across versions. Neither the promises nor the prospective product capabilities are verified by this preference label.

Review tags: preference_ambiguity, unsupported_additions. Flags: stable mismatch in both orders. Released individual preference strengths: [1, 1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-8953 — Career-search email and call to action

Publisher label: **A**. Baseline **A/B**; 20 updates **A/B**; 200 updates **A/B**.

Both use the requested second-place theme, address the community, and ask readers to book a discovery call. A lists more call benefits; B is more compact and repeats the slogan in a postscript. Both-order first-position selection persists across all three versions.

Review tags: preference_ambiguity, overlapping_content. Flags: order disagreement. Released individual preference strengths: [-1, -1, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9013 — Same nine forest percentages

Publisher label: **tie_preference**. Baseline **tie/B**; 20 updates **tie/tie**; 200 updates **tie/B**.

Both answers contain exactly the same nine requested percentages, in the same year and country order, and both use bulleted lists. The supplied reference text supports those values; no claim is made about the live site's current data. Bullet style and punctuation differ. After 200 updates the original choice is tie, but it becomes B when B is displayed first.

Review tags: input_check, format_only_difference. Flags: swapped-order regression from 20 updates, order disagreement. Released individual preference strengths: [-1, -1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9016 — Five-day vegetarian plan under 200 words

Publisher label: **A**. Baseline **A/B**; 20 updates **A/B**; 200 updates **A/B**.

A has 108 whitespace-separated words and headings for all five days. B has 364 words, missing early day headings and a stray markup fragment. B visibly exceeds the requested 200-word maximum. Nutritional suitability was not assessed. The model nevertheless selects B when B appears first, in all three versions.

Review tags: input_check, constraint_mismatch. Flags: order disagreement. Released individual preference strengths: [-2, -3, 1, 2, -3].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9019 — How many nieces?

Publisher label: **B**. Baseline **tie/B**; 20 updates **tie/B**; 200 updates **tie/B**.

Both decline an exact niece count because relevant information is missing. A's explanation omits Harry's seven children when it says there are four nieces/nephews in total; the stated counts sum to eleven, before any unknown children of Keisha. B retains all siblings and asks about the missing information. The final tie/B pair overlooks that difference in original order.

Review tags: input_check, reasoning_error. Flags: order disagreement. Released individual preference strengths: [3, 1, -3, 2, 3].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9024 — Fine-structure constant

Publisher label: **A**. Baseline **B/A**; 20 updates **B/A**; 200 updates **B/A**.

The answers make materially different physics claims, including B's assertion that the constant puts an upper bound on light speed. Those claims require specialist/source checking, not a new label from this review. Every model version selects the displayed second answer, yielding B/A after mapping back.

Review tags: external_facts_unchecked, human_adjudication_needed. Flags: order disagreement. Released individual preference strengths: [2, -3, 1, -2, -2].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9040 — Professional baseball history

Publisher label: **B**. Baseline **B/A**; 20 updates **B/B**; 200 updates **B/A**.

A lists eight franchises and B focuses on two; both include many precise historical and championship claims without supplied sources. Those dates, continuity claims, and totals were not verified. The 20-update stable B preference becomes B/A at 200, meaning the final model selects the second displayed answer in either order.

Review tags: external_facts_unchecked, human_adjudication_needed. Flags: swapped-order regression from 20 updates, order disagreement. Released individual preference strengths: [2, 1, -2, 3, 3].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9042 — LEI explanation around 400 words

Publisher label: **tie_preference**. Baseline **A/B**; 20 updates **A/B**; 200 updates **A/B**.

Both address definition, adoption, and benefits, at 440 and 458 whitespace-separated words. The prompt's approximate length therefore does not provide a clear discriminator. Regulatory and benefit claims were not checked. The model chooses the displayed first answer in both orders across all versions.

Review tags: preference_ambiguity, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [1, -1, 1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9053 — More complex Discord UI

Publisher label: **B**. Baseline **A/B**; 20 updates **A/B**; 200 updates **A/B**.

The latest question asks about UI options in the context of a Python bot. A lists bot features and integration approaches; B lists language libraries and direct API access. Neither is an executed or verified solution, and current library/UI capabilities were not checked. The model repeatedly favors whichever answer is first.

Review tags: coverage_difference, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [2, 1, 1, 2, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9057 — SLF4J Logger explanation

Publisher label: **B**. Baseline **A/B**; 20 updates **tie/B**; 200 updates **A/B**.

A emphasizes the logging abstraction and switching implementations; B focuses on levels, exceptions, and stack traces. Both respond to the latest question more directly than the unrelated earlier chat history. API and class/interface terminology were not verified. The final model picks the first displayed answer.

Review tags: coverage_difference, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [-1, -2, 2, 1, 2].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9077 — Hamlet acknowledgment with a suspect premise

Publisher label: **tie_preference**. Baseline **B/B**; 20 updates **tie/B**; 200 updates **B/B**.

The last user turn is a thank-you. A acknowledges it; B repeats an expected-marriage claim about Ophelia and Laertes. The supplied user history calls them siblings, and the Folger edition explicitly identifies Ophelia as Laertes's sister. That checks the sibling relationship, not every historical claim in the exchange. The publisher tie has conflicting reviewer preferences, so a label match would not certify B's extra claim.

Review tags: source_checked_context, human_adjudication_needed. Flags: original-order regression from 20 updates. Released individual preference strengths: [3, 1, 1, 3, -1].

Label unchanged. No replacement label. Not eligible for training.

## helpsteer2-preference-9104 — Simple MIS explanation

Publisher label: **A**. Baseline **A/B**; 20 updates **A/B**; 200 updates **A/B**.

A separately explains the service cycle and its analysis but is lengthy; B is shorter and uses a different stage list. The user asks for a simple MIS explanation. No course-specific reference establishes a unique correct stage list here. The model selects the first displayed answer across all three versions.

Review tags: coverage_difference, external_facts_unchecked. Flags: order disagreement. Released individual preference strengths: [2, -1, 2, -2, -1].

Label unchanged. No replacement label. Not eligible for training.

Source text: [frozen 60-question sample](frozen_baseline_v1/sample.jsonl). Find the full ID in that file. [Machine-readable reviews](results/error_review_20261006T221309230086Z/case_reviews.jsonl) preserve all fields. Source checks for Redis and Hamlet are linked in [the main report](ERROR_REVIEW.md).
