# Assistant-prepared evidence drafts — 50 FiQA cases

**Not 50 completed human reviews.** These are an AI-assisted evidence summary for fixed HNSW `efSearch=10` failures, **not deployed CURA failures**. The instructor's manual taxonomy requirement still needs a person to inspect all 50 original passages and record/confirm each judgment. No answers are inserted into human-review worksheet fields, and no case is marked complete. Do not use these evaluated queries for tuning CURA.

The *observed failure* below follows directly from relevance labels and nDCG; possible reasons for the graph search error are hypotheses, not established mechanisms.

## Case 1 — query 10845

**Question:** Rationale behind using 12, 26 and 9 to calculate MACD

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `191588` (relevance `1`) says: “The values of 12, 26 and 9 are the typical industry standard setting used with the MACD, however other values can be substituted depending on your trading style and goals. The 26d EMA is considered the long moving average when in this case it is compared to the shorter 12d EMA. I”.

**Exact top-1:** Document `191588` (relevance `1`) — “The values of 12, 26 and 9 are the typical industry standard setting used with the MACD, however other values can be substituted depending on your trading style and goals. The 26d EMA is considered the long moving average when in this case it is compared to the shorter 12d EMA. I”.

**HNSW ef=10 top-1:** Document `316132` (relevance `0`) — “"Of course it can. This is a time value of money calculation. If I knew the maturity date, or current yield to maturity I'd be able to calculate the other number and advise how much rates need to rise to cause the value to drop from 18 to 17. For a 10 year bond, a rise today of .”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.60`.

**Assistant draft analysis:** The judged-relevant passage addresses “Rationale behind using 12, 26 and 9 to calculate MACD”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 2 — query 1451

**Question:** How do you find an ethical, honest independent insurance broker in Canada?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `538005` (relevance `1`) says: “How do you find an ethical, honest practitioner of any business? One: Make a small transaction with them and see how they treat you. If they cheat you on something small, don't give them a chance with something big. Two: Ask family and friends for recommendations. Three: Get info”.

**Exact top-1:** Document `538005` (relevance `1`) — “How do you find an ethical, honest practitioner of any business? One: Make a small transaction with them and see how they treat you. If they cheat you on something small, don't give them a chance with something big. Two: Ask family and friends for recommendations. Three: Get info”.

**HNSW ef=10 top-1:** Document `533621` (relevance `0`) — “"Credit scores are not such a big deal in Canada as they are in the US and even some European countries. One reason for this: the Social Insurance Number (SIN number) isn't used for so many purposes like the Social Security Number (SSN) in the US. The SIN number isn't even requir”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.20`.

**Assistant draft analysis:** The judged-relevant passage addresses “How do you find an ethical, honest independent insurance broker in Canada?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 3 — query 4785

**Question:** What is the difference between a structured collar and a normal collar in finance?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `37517` (relevance `1`) says: “Let's start with a definition: A Collar is a protective strategy for a position in the underlying instrument created by purchasing a put and selling a call to partially pay for the put option purchased or vice versa. Based on that definition, there are two different types of coll”.

**Exact top-1:** Document `37517` (relevance `1`) — “Let's start with a definition: A Collar is a protective strategy for a position in the underlying instrument created by purchasing a put and selling a call to partially pay for the put option purchased or vice versa. Based on that definition, there are two different types of coll”.

**HNSW ef=10 top-1:** Document `229582` (relevance `0`) — “I am not a structured products expert, just relaying what's in the article, but I'd imagine it includes a larger equity tranche in the structured product, thereby giving more cushion against losses to the actual debt investors in the tranches above it in the payment water fall.”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.20`.

**Assistant draft analysis:** The judged-relevant passage addresses “What is the difference between a structured collar and a normal collar in finance?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 4 — query 5592

**Question:** Are “hard money loans” meant only for real estate?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `355241` (relevance `1`) says: “From Wikipedia: A hard money loan is a specific type of asset-based loan financing through which a borrower receives funds secured by the value of a parcel of real estate. Hard money loans are typically issued at much higher interest rates than conventional commercial or resident”.

**Exact top-1:** Document `355241` (relevance `1`) — “From Wikipedia: A hard money loan is a specific type of asset-based loan financing through which a borrower receives funds secured by the value of a parcel of real estate. Hard money loans are typically issued at much higher interest rates than conventional commercial or resident”.

**HNSW ef=10 top-1:** Document `416617` (relevance `0`) — “Real loans are forgone consumption, like a savings account. In real terms, you can loan only when you put aside something from consuming it. Any kind of monetary tricks and expansion cant create real savings out of thin air, only an illusion of savings like in economic bubbles: t”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “Are “hard money loans” meant only for real estate?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 5 — query 721

**Question:** What are “upstream investments” and “downstream investments” in this context?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `496225` (relevance `1`) says: “Upstream is into businesses that supply the original business; downstream is into businesses that make use of the original product. So in that description, what they are saying is that the original business received products from plantations and sent products to manufacturing. Th”.

**Exact top-1:** Document `496225` (relevance `1`) — “Upstream is into businesses that supply the original business; downstream is into businesses that make use of the original product. So in that description, what they are saying is that the original business received products from plantations and sent products to manufacturing. Th”.

**HNSW ef=10 top-1:** Document `589088` (relevance `0`) — “"Some of the other answers recommended peer-to-peer lending and property markets. I would not invest in either of these. Firstly, peer-to-peer lending is not a traditional investment and we may not have enough historical data for the risk-to-return ratio. Secondly, property inves”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.60`.

**Assistant draft analysis:** The judged-relevant passage addresses “What are “upstream investments” and “downstream investments” in this context?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 6 — query 8795

**Question:** Stock exchanges using open outcry

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `57297` (relevance `1`) says: “As Chris pointed out in his comment, smaller stock exchanges may use open outcry. There are several exchanges that use open outcry/floor trading in the US, however, although they aren't necessarily stock exchanges. Having visited the three Chicago exchanges I mentioned, I can per”.

**Exact top-1:** Document `57297` (relevance `1`) — “As Chris pointed out in his comment, smaller stock exchanges may use open outcry. There are several exchanges that use open outcry/floor trading in the US, however, although they aren't necessarily stock exchanges. Having visited the three Chicago exchanges I mentioned, I can per”.

**HNSW ef=10 top-1:** Document `68907` (relevance `0`) — “I'd add, this is actually the way any stock opens every day, i.e. the closing price of the prior day is what it is, but the opening price will reflect whatever news there was prior to the day's open. If you watch the business news, you'll often see that some stock has an order im”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “Stock exchanges using open outcry”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 7 — query 904

**Question:** How do I set up Quickbooks for a small property rental company that holds its properties in separate LLC's?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `415514` (relevance `1`) says: “"You need one ""company file"" for each company that you want to track through QuickBooks. Looks like, in your case, that is at least the PM and the PH (as you labeled them in your question). The companies that just hold property and pay utilities might be simple enough that you”.

**Exact top-1:** Document `415514` (relevance `1`) — “"You need one ""company file"" for each company that you want to track through QuickBooks. Looks like, in your case, that is at least the PM and the PH (as you labeled them in your question). The companies that just hold property and pay utilities might be simple enough that you”.

**HNSW ef=10 top-1:** Document `224438` (relevance `0`) — “for starters get a cheap easy accounting software pack like quickbooks and have the salesmen train you on it's use and set-up. the 50k you put into the company will count as paid up share capital. then any future withdrawal from company account will show as loan to director.”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.30`.

**Assistant draft analysis:** The judged-relevant passage addresses “How do I set up Quickbooks for a small property rental company that holds its properties in separate LLC's?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 8 — query 9824

**Question:** Where can end-of-day data be downloaded for corporate bonds?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `574777` (relevance `1`) says: “Here is one from a Bloomberg partnership, it is free. To get the end of day prices, you may need some programming done. PM me if you need help with that. Getting bond quotes and general information about a bond issue is considerably more difficult than researching a stock or a mu”.

**Exact top-1:** Document `574777` (relevance `1`) — “Here is one from a Bloomberg partnership, it is free. To get the end of day prices, you may need some programming done. PM me if you need help with that. Getting bond quotes and general information about a bond issue is considerably more difficult than researching a stock or a mu”.

**HNSW ef=10 top-1:** Document `176640` (relevance `0`) — “A pencil and a small notepad really work here, but if you have a smartphone then some way of using it makes sense as well. Try: Transcribe all of these onto a better record at the end of each day. Also record the amount of money in your wallet/purse/pocket every day, and check to”.

**Measured discrepancy:** Exact nDCG@10 `1.0000`, HNSW nDCG@10 `0.0000`, nDCG loss `1.0000`, ANN recall@10 `0.20`.

**Assistant draft analysis:** The judged-relevant passage addresses “Where can end-of-day data be downloaded for corporate bonds?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 9 — query 11088

**Question:** Am I required to have a lawyer create / oversee creation of my will?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `437100` (relevance `1`) says: “"This is not intended as legal advice, and only covers general knowledge I have on the subject of wills as a result of handling my own finances. Each state of the USA has its own laws on wills and trusts. You can find these online. For example, in Kentucky I found state laws here”.

**Exact top-1:** Document `295142` (relevance `0`) — “"**What you need to know about [estate planning](http://money.cnn.com/magazines/moneymag/money101/lesson21/), including why you may need a will and assigning a power of attorney.** **1. No matter your net worth, it's important to have a basic estate plan in place.** Such a plan e”.

**HNSW ef=10 top-1:** Document `330622` (relevance `0`) — “Delaware LLC requires that each business entity have and hold an enterprise Registered in the State of Delaware who can be both a character resident or enterprise entity this is legal to do business in the Wilmington, Delaware. the Delaware LLC has offered the same asset protecti”.

**Measured discrepancy:** Exact nDCG@10 `0.6309`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6309`, ANN recall@10 `0.20`.

**Assistant draft analysis:** The judged-relevant passage addresses “Am I required to have a lawyer create / oversee creation of my will?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 10 — query 1198

**Question:** What are the consequences of IRS “reclassification” on both employer and employee?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `56718` (relevance `1`) says: “"You are confusing entirely unrelated things. First the ""profit distribution"" issue with Bob's S-Corp which is in fact tax evasion and will probably trigger a very nasty audit. Generally, if you're the sole employee of your own S-Corp, and the whole S-Corp income is from your o”.

**Exact top-1:** Document `439470` (relevance `0`) — “As well as the visual impact - employees don't see the money disappearing directly from their paycheck so don't feel as heavily taxed - there's usually one practical difference, at least in the UK. Employers are generally prohibited in law from explicitly passing the tax onto the”.

**HNSW ef=10 top-1:** Document `439470` (relevance `0`) — “As well as the visual impact - employees don't see the money disappearing directly from their paycheck so don't feel as heavily taxed - there's usually one practical difference, at least in the UK. Employers are generally prohibited in law from explicitly passing the tax onto the”.

**Measured discrepancy:** Exact nDCG@10 `0.6309`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6309`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “What are the consequences of IRS “reclassification” on both employer and employee?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 11 — query 4946

**Question:** Something looks off about Mitsubishi financial data

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `121690` (relevance `1`) says: “All but certainly, Mitsubishi is selling so cheaply because of the fuel scandal. It has been providing false fuel efficiency data for decades. As a result, it may face significant penalties and may have lost the trust of consumers, who will now be less likely to purchase a Mitsub”.

**Exact top-1:** Document `564338` (relevance `0`) — “http://www.pacificrubiales.com/investor-relations/reports.html does have financial reports on their website for the example you list. There is the potential for some data to not be easily imported into a format that Yahoo! Finance uses would be my guess for why some data may be m”.

**HNSW ef=10 top-1:** Document `564338` (relevance `0`) — “http://www.pacificrubiales.com/investor-relations/reports.html does have financial reports on their website for the example you list. There is the potential for some data to not be easily imported into a format that Yahoo! Finance uses would be my guess for why some data may be m”.

**Measured discrepancy:** Exact nDCG@10 `0.6309`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6309`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “Something looks off about Mitsubishi financial data”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 12 — query 5125

**Question:** Regarding Australian CBS takeover of TEN

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `318728` (relevance `1`) says: “"they are purchasing the company"" is this correct? Yes this is correct. If I purchase a ""company"" here in Australia, I also purchase its assets and liabilities Yes that is correct. How can it be NIL? How can it be legal? The value of shares [or shareholders] is Assets - liabil”.

**Exact top-1:** Document `73457` (relevance `0`) — “"For non Australian residents: Dividends withholding tax rate is 30%. Depending upon your country of residence where there is a tax treaty in place to avoid double taxation, then this can be reduced. Note that only dividends that are unfranked are subject to this (in Australia, i”.

**HNSW ef=10 top-1:** Document `319793` (relevance `0`) — “You talk about an individual not being advised to sell (or purchase) in response to trends in the market in such a buy and hold strategy. But think of this for a moment: You buy stock ABC for $10 when both the market as a whole and stock ABC are near the bottom of a bear market a”.

**Measured discrepancy:** Exact nDCG@10 `0.6309`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6309`, ANN recall@10 `0.20`.

**Assistant draft analysis:** The judged-relevant passage addresses “Regarding Australian CBS takeover of TEN”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 13 — query 7105

**Question:** What is the difference between fixed-income duration and equity duration?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `72652` (relevance `1`) says: “A bond has a duration that can be easily calculated. It's the time weighted average of all the payments you'll receive and helpful to understand the effect a change in rates will have on that instrument. The duration of a stock, on the other hand, is a forced construct to then us”.

**Exact top-1:** Document `496096` (relevance `0`) — “"The fact that dividends grow in perpetuity does not prevent one from calculating duration. In fact, many academic papers look at exactly this problem, such as Lewin and Satchell. This Wilmott thread discusses some of the pros and cons of the concept in some detail. PS: Although”.

**HNSW ef=10 top-1:** Document `400747` (relevance `0`) — “The changes to Equity given are: Since the total change is 42,500, the difference would be change in Retained Earnings (net income), so net income is”.

**Measured discrepancy:** Exact nDCG@10 `0.6309`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6309`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “What is the difference between fixed-income duration and equity duration?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 14 — query 10034

**Question:** Tax implications of holding EWU (or other such UK ETFs) as a US citizen?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `181942` (relevance `1`) says: “My understanding is that EWU (and EWUS) are both traded on US stock markets (NYSE & BATS), and as such these are not classified as PFIC. However, they do contain PFICs, so iShares takes the responsibility of handling the PFICs they contain and make adjustments in December. This c”.

**Exact top-1:** Document `181942` (relevance `1`) — “My understanding is that EWU (and EWUS) are both traded on US stock markets (NYSE & BATS), and as such these are not classified as PFIC. However, they do contain PFICs, so iShares takes the responsibility of handling the PFICs they contain and make adjustments in December. This c”.

**HNSW ef=10 top-1:** Document `569210` (relevance `0`) — “if you open spreadbetting accounts and prove to the inland revenue that trading profits are not your main income, you will not be liable for any tax on your gains. Holding a property in the uk which is point of call for any held bank accounts needs to be verifiable though. This i”.

**Measured discrepancy:** Exact nDCG@10 `0.6131`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6131`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “Tax implications of holding EWU (or other such UK ETFs) as a US citizen?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 15 — query 2316

**Question:** What exchange rate does El Al use when converting final payment amount to shekels?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `348955` (relevance `1`) says: “In older days the merchants and their merchant banks[or service providers] would take funds in their currency. Say in this case USD. When the charge hits the issuer bank, the merchant and merchant bank gets there USD and were happy. The user would get charged in local currency Sh”.

**Exact top-1:** Document `348955` (relevance `1`) — “In older days the merchants and their merchant banks[or service providers] would take funds in their currency. Say in this case USD. When the charge hits the issuer bank, the merchant and merchant bank gets there USD and were happy. The user would get charged in local currency Sh”.

**HNSW ef=10 top-1:** Document `262589` (relevance `0`) — “"I don't see EWQ6 in any of your links, so I can't say for certain, but when you buy an option contract on a future, the option will be for a specific future (and strike). So the page you're looking at may be for options on E-mini S&P 500 futures in general, and when you actually”.

**Measured discrepancy:** Exact nDCG@10 `0.6131`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6131`, ANN recall@10 `0.00`.

**Assistant draft analysis:** The judged-relevant passage addresses “What exchange rate does El Al use when converting final payment amount to shekels?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 16 — query 3085

**Question:** How long can I convert 401(k) to Roth 401(k)?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `248619` (relevance `1`) says: “Conversions must be done during the calendar year. This would apply to both IRA and 401(k) accounts. For IRAs, deposits may be made until 4/15, and the same holds for Solo 401(k) accounts. For conversions, the IRA permits a recharacterization, basically, a do-over, which reverses”.

**Exact top-1:** Document `248619` (relevance `1`) — “Conversions must be done during the calendar year. This would apply to both IRA and 401(k) accounts. For IRAs, deposits may be made until 4/15, and the same holds for Solo 401(k) accounts. For conversions, the IRA permits a recharacterization, basically, a do-over, which reverses”.

**HNSW ef=10 top-1:** Document `380047` (relevance `0`) — “The plan is perfectly valid and legal as tax rules currently stand. There is no limit to the amount you can rollover or convert to Roth. Assuming your 401(k) is traditional pre-tax, you'll have to pay income taxes on the amount you convert above personal exemption + standard dedu”.

**Measured discrepancy:** Exact nDCG@10 `0.6131`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6131`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “How long can I convert 401(k) to Roth 401(k)?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 17 — query 594

**Question:** Should a retail trader bother about reading SEC filings

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `377322` (relevance `1`) says: “"There are many different kinds of SEC filings with different purposes. Broadly speaking, what they have in common is that they are the ways that companies publicly disclose information that they are legally required to disclose. The page that you listed gives brief descriptions”.

**Exact top-1:** Document `377322` (relevance `1`) — “"There are many different kinds of SEC filings with different purposes. Broadly speaking, what they have in common is that they are the ways that companies publicly disclose information that they are legally required to disclose. The page that you listed gives brief descriptions”.

**HNSW ef=10 top-1:** Document `527105` (relevance `0`) — “I would echo @Victor's comments. One book and 1000 web pages doesnt make you a good investor/trader. There are some basic things you should be aware of and read up on There are a few books that I would recommend I have been trading for over 10 years, my dad for over 30 years and”.

**Measured discrepancy:** Exact nDCG@10 `0.6131`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6131`, ANN recall@10 `0.30`.

**Assistant draft analysis:** The judged-relevant passage addresses “Should a retail trader bother about reading SEC filings”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 18 — query 8

**Question:** How to deposit a cheque issued to an associate in my business into my business account?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `566392` (relevance `1`) says: “Have the check reissued to the proper payee.”.

**Exact top-1:** Document `65404` (relevance `1`) — “Just have the associate sign the back and then deposit it. It's called a third party cheque and is perfectly legal. I wouldn't be surprised if it has a longer hold period and, as always, you don't get the money if the cheque doesn't clear. Now, you may have problems if it's a lar”.

**HNSW ef=10 top-1:** Document `499889` (relevance `0`) — “I have no idea what the traditional accounting way of dealing with this might be; but does your accounts package has the concept of subaccounts within a bank account? If so, to me it would make sense that when a cheque is written, you move money in the accounts package from the b”.

**Measured discrepancy:** Exact nDCG@10 `0.6131`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6131`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “How to deposit a cheque issued to an associate in my business into my business account?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 19 — query 8271

**Question:** Income in zero-interest environment

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `376709` (relevance `1`) says: “"anything that produces steady income will produce a ""real return"" (return above inflation) in a zero-interest rate environment: Note, however, that all of these will decline in value if interest rates rise."”.

**Exact top-1:** Document `376709` (relevance `1`) — “"anything that produces steady income will produce a ""real return"" (return above inflation) in a zero-interest rate environment: Note, however, that all of these will decline in value if interest rates rise."”.

**HNSW ef=10 top-1:** Document `136262` (relevance `0`) — “I got that notion from Max Kaiser, his ideas about that are mostly about interest rate apartheid - banks and rich people get money at zero interest, we pay 9, 10, 30%. I think it involves everything from how we are seen in the eyes of the law to assuming risk, etc. We are expecte”.

**Measured discrepancy:** Exact nDCG@10 `0.6131`, HNSW nDCG@10 `0.0000`, nDCG loss `0.6131`, ANN recall@10 `0.30`.

**Assistant draft analysis:** The judged-relevant passage addresses “Income in zero-interest environment”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 20 — query 4681

**Question:** How to fix Finance::Quote to pull quotes in GnuCash

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `181909` (relevance `1`) says: “"The Yahoo Finance API is no longer available, so Finance::Quote needs to point at something else. Recent versions of Finance::Quote can use AlphaVantage as a replacement for the Yahoo Finance API, but individual users need to acquire and input an AlphaVantage API key. Pretty dec”.

**Exact top-1:** Document `210707` (relevance `0`) — “The best way to answer this question is to try. GnuCash is free, so setting it up and giving it a go shouldn't be too hard. After all, what really matters is how helpful the program is for your purposes. One aspect of personal finance that stops me from jumping to GnuCash/KMyMone”.

**HNSW ef=10 top-1:** Document `221040` (relevance `0`) — “"&gt; ...those cross-party votes weren't going to change the result, unlike in your examples. You make a fair case but I still don't agree this is justification for McConnell's inaction. &gt; If they had somehow, someway, gotten even a tiny bit of Republican buy-in, we'd be havin”.

**Measured discrepancy:** Exact nDCG@10 `0.5250`, HNSW nDCG@10 `0.0000`, nDCG loss `0.5250`, ANN recall@10 `0.00`.

**Assistant draft analysis:** The judged-relevant passage addresses “How to fix Finance::Quote to pull quotes in GnuCash”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 21 — query 7590

**Question:** Why are US target retirement funds weighted so heavily towards US stocks?

**Suggested category (requires human confirmation):** 2 — Lower ranking utility than exact retrieval (some relevance survives)

**Relevant evidence:** Judged-relevant document `105165` (relevance `1`) says: “"Excellent question, though any why question can be challenging to answer because it depends on the financial products in question. At least, I haven't seen many target date retirement funds that include a high percent of foreign stocks, so below explains the ones I've seen which”.

**Exact top-1:** Document `105165` (relevance `1`) — “"Excellent question, though any why question can be challenging to answer because it depends on the financial products in question. At least, I haven't seen many target date retirement funds that include a high percent of foreign stocks, so below explains the ones I've seen which”.

**HNSW ef=10 top-1:** Document `569834` (relevance `0`) — “"To expand a bit on @MSalters's answer ... When I read your question title I assumed that by ""retirement funds"" you meant target-date funds that are close to their target dates (say, the 2015 target fund). When I saw that you were referring to all target-date funds, it occurred”.

**Measured discrepancy:** Exact nDCG@10 `0.8772`, HNSW nDCG@10 `0.3869`, nDCG loss `0.4904`, ANN recall@10 `0.60`.

**Assistant draft analysis:** The judged-relevant passage addresses “Why are US target retirement funds weighted so heavily towards US stocks?”, whereas the HNSW top-ranked passage begins with the different material quoted above. The ef=10 ranking retained some relevance but lost nDCG relative to the exact nearest-neighbor ranking. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 22 — query 2400

**Question:** Will I be paid dividends if I own shares?

**Suggested category (requires human confirmation):** 2 — Lower ranking utility than exact retrieval (some relevance survives)

**Relevant evidence:** Judged-relevant document `1198` (relevance `1`) says: “Yes, as long as you own the shares before the ex-dividend date you will get the dividends. Depending on your instructions to your broker, you can receive cash dividends or you can have the dividends reinvested in more shares of the company. There are specific Dividend ReInvestmen”.

**Exact top-1:** Document `1198` (relevance `1`) — “Yes, as long as you own the shares before the ex-dividend date you will get the dividends. Depending on your instructions to your broker, you can receive cash dividends or you can have the dividends reinvested in more shares of the company. There are specific Dividend ReInvestmen”.

**HNSW ef=10 top-1:** Document `365627` (relevance `0`) — “"If a stock is trading for $11 per share just before a $1 per share dividend is declared, then the share price drops to $10 per share immediately following the declaration. If you owned 100 shares (valued at $1100) before the dividend was declared, then you still own 100 shares (”.

**Measured discrepancy:** Exact nDCG@10 `0.6257`, HNSW nDCG@10 `0.1672`, nDCG loss `0.4585`, ANN recall@10 `0.90`.

**Assistant draft analysis:** The judged-relevant passage addresses “Will I be paid dividends if I own shares?”, whereas the HNSW top-ranked passage begins with the different material quoted above. The ef=10 ranking retained some relevance but lost nDCG relative to the exact nearest-neighbor ranking. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 23 — query 1297

**Question:** Why aren't there solutions for electronic itemized receipt for retail in-store purchases?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `171761` (relevance `1`) says: “In some stores that is done. When I shop at the Apple store or at the Farmers market the receipt is automatically sent to my email address. Why don't others do it? If the target of the itemized receipt is a credit card company they would be sending data that they spent collecting”.

**Exact top-1:** Document `46658` (relevance `0`) — “The stores track the individual items for inventory planning and marketing purposes. Having worked in the transaction processing business for a time (writing one), I can say with confidence that the credit/debit card companies do not receive an itemized list of the items involved”.

**HNSW ef=10 top-1:** Document `270540` (relevance `0`) — “Scanning receipts is easy and any decent scanner will do a good job for you. The difficult part is the software that 'extracts' the data. Today there is no software that can do this really well because there is just too great a range of receipts (e.g. handwritten receipts, receip”.

**Measured discrepancy:** Exact nDCG@10 `0.4307`, HNSW nDCG@10 `0.0000`, nDCG loss `0.4307`, ANN recall@10 `0.40`.

**Assistant draft analysis:** The judged-relevant passage addresses “Why aren't there solutions for electronic itemized receipt for retail in-store purchases?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 24 — query 6985

**Question:** Bed and Breakfast, Same Day Capital Gains UK

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `456436` (relevance `1`) says: “The 'same day rule' in the UK is a rule for matching purposes only. It says that sales on any day are matched firstly with purchases made on the same day for the purposes of ascertaining any gain/loss. Hence the phrase 'bed-and-breakfast' ('b&b') when you wish to crystalise a gai”.

**Exact top-1:** Document `444405` (relevance `0`) — “Here's how capital gains are totaled: Long and Short Term. Capital gains and losses are either long-term or short-term. It depends on how long the taxpayer holds the property. If the taxpayer holds it for one year or less, the gain or loss is short-term. Net Capital Gain. If a ta”.

**HNSW ef=10 top-1:** Document `444405` (relevance `0`) — “Here's how capital gains are totaled: Long and Short Term. Capital gains and losses are either long-term or short-term. It depends on how long the taxpayer holds the property. If the taxpayer holds it for one year or less, the gain or loss is short-term. Net Capital Gain. If a ta”.

**Measured discrepancy:** Exact nDCG@10 `0.4307`, HNSW nDCG@10 `0.0000`, nDCG loss `0.4307`, ANN recall@10 `0.90`.

**Assistant draft analysis:** The judged-relevant passage addresses “Bed and Breakfast, Same Day Capital Gains UK”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 25 — query 4007

**Question:** What is a reasonable salary for the owner and sole member of a small S-Corp?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `388704` (relevance `1`) says: “"Generally if you're a sole S-Corp employee - it is hard to explain how the S-Corp earned more money than your work is worth. So it is reasonable that all the S-Corp profits would be pouring into your salary. Especially when the amounts are below the FICA SS limits when separatin”.

**Exact top-1:** Document `559900` (relevance `0`) — “I think it is clear that this has a lot to do with keeping salaries down. The company has a maximal possible salary limit (i.e. the most they would be willing and able to pay) for what you do. You have a minimal possible salary. Both of you know that the current salary is in betw”.

**HNSW ef=10 top-1:** Document `346374` (relevance `0`) — “Compared with a Sole Proprietorship, the main disadvantages of an S-Corporation or an LLC are that it adds a lot of management overhead (time, and possibly money if you don't do it all yourself), and there are fees you must pay to incorporate, as well as additional yearly mainten”.

**Measured discrepancy:** Exact nDCG@10 `0.4018`, HNSW nDCG@10 `0.0000`, nDCG loss `0.4018`, ANN recall@10 `0.40`.

**Assistant draft analysis:** The judged-relevant passage addresses “What is a reasonable salary for the owner and sole member of a small S-Corp?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 26 — query 4999

**Question:** Looking for a good source for Financial Statements

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `314898` (relevance `1`) says: “If you're researching a publicly traded company in the USA, you can search the company filings with the SEC. Clicking 'Filings' should take you here.”.

**Exact top-1:** Document `9938` (relevance `1`) — “The best source of financial statements would be from the company in question. On corporate websites of public listed companies, you can find such financial statements uploaded in the Investor's Relations section of their website. If their company does not have an online presence”.

**HNSW ef=10 top-1:** Document `4153` (relevance `0`) — “Congratulations on being in such good financial state. You have a few investment choices. If you want very low risk, you are talking bonds or CDs. With the prime rate so low, nobody is paying anything useful for very low risk investments. However, my opinion is that given your fi”.

**Measured discrepancy:** Exact nDCG@10 `0.3904`, HNSW nDCG@10 `0.0000`, nDCG loss `0.3904`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “Looking for a good source for Financial Statements”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 27 — query 3830

**Question:** US citizen transferring money to Indian fiance to buy property

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `54952` (relevance `1`) says: “A. Kindly avoid taking dollars in form of cash to india unless and until it is an emergency. Once the dollar value is in excess of $10,000, you need to declare the same with Indian customs at the destination. Even though it is not a cumbersome procedure, why unnecessarily undergo”.

**Exact top-1:** Document `414772` (relevance `0`) — “Buyer A didn't send money to the US government, Buyer A sent money to Seller B, a US resident. I think the most common way to facilitate a transaction like this is a regular old international wire transfer. Buyer A in India goes to their bank to exchange X INR to $1mm USD. $1mm U”.

**HNSW ef=10 top-1:** Document `414772` (relevance `0`) — “Buyer A didn't send money to the US government, Buyer A sent money to Seller B, a US resident. I think the most common way to facilitate a transaction like this is a regular old international wire transfer. Buyer A in India goes to their bank to exchange X INR to $1mm USD. $1mm U”.

**Measured discrepancy:** Exact nDCG@10 `0.3562`, HNSW nDCG@10 `0.0000`, nDCG loss `0.3562`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “US citizen transferring money to Indian fiance to buy property”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 28 — query 5981

**Question:** Is it a good investment for a foreigner to purchase a flat/apartment in China?

**Suggested category (requires human confirmation):** 2 — Lower ranking utility than exact retrieval (some relevance survives)

**Relevant evidence:** Judged-relevant document `171483` (relevance `1`) says: “No, it's not. This could be a great question, but with no background, not so much. Do you live there now? For how long, and how much longer? You say investment, are you looking to live in it or rent it out? I have nothing against China, but I'd not buy anywhere unless the price,”.

**Exact top-1:** Document `286182` (relevance `1`) — “It is a lousy investment to purchase an apartment in China. Chinese citizens purchase apartments in China because, well... here's how China works: There's some fundamentals driving Chinese property values higher, but mostly it's a bubble caused by those reasons.”.

**HNSW ef=10 top-1:** Document `103333` (relevance `0`) — “Perhaps I can: my city is full of towers of Chinese investor owned condos that were purchased to gain EB5 visas. These visas require a $500k investment in the US, in an economically depressed area. The census tracts were gerrymandered together to make a nice part of town qualify.”.

**Measured discrepancy:** Exact nDCG@10 `0.4852`, HNSW nDCG@10 `0.1696`, nDCG loss `0.3156`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “Is it a good investment for a foreigner to purchase a flat/apartment in China?”, whereas the HNSW top-ranked passage begins with the different material quoted above. The ef=10 ranking retained some relevance but lost nDCG relative to the exact nearest-neighbor ranking. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 29 — query 4700

**Question:** Better to get loan from finance company or bank considering the drop of credit score?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `345137` (relevance `1`) says: “If your primary concern is a drop in your credit score, go to a mortgage broker instead of multiple banks and finance companies. Each time you ask a bank or financial institution for a loan, they do a hard pull on your credit rating which costs you a couple of points. Visit a doz”.

**Exact top-1:** Document `136040` (relevance `0`) — “No. Getting more credit lowers your credit utilization ratio (if you don't use it), which raises your credit rating, this can also be done by asking for a higher limit on your existing credit card. Also, there is a chance that the company you got your first card from won't pull y”.

**HNSW ef=10 top-1:** Document `27429` (relevance `0`) — “"Because the USA is the world's biggest economy - everybody in the world works with the USA (even if the american companies are not direct suppliers, they are surely somewhere in the supply chain). If USA credit rating is lower, that means american companies will find it harder t”.

**Measured discrepancy:** Exact nDCG@10 `0.3155`, HNSW nDCG@10 `0.0000`, nDCG loss `0.3155`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “Better to get loan from finance company or bank considering the drop of credit score?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 30 — query 9644

**Question:** If early exercise is a bad idea, why American option is more expensive than European [duplicate]

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `194605` (relevance `1`) says: “There are a few situations in which it may be advantageous to exercise early. Wikipedia actually has a good explanation: Option Style, Difference in value To account for the American's higher value there must be some situations in which it is optimal to exercise the American opti”.

**Exact top-1:** Document `364575` (relevance `0`) — “OK, my fault for not doing more research. Wikipedia explains this well: http://en.wikipedia.org/wiki/Option_style#Difference_in_value Basically, there are some cases where it's advantageous to exercise an American option early. For non-gold currency options, this is only when the”.

**HNSW ef=10 top-1:** Document `396662` (relevance `0`) — “&gt;If everyone in auto manufacturing labor was replaced by a cheap robot tomorrow, then cars would be cheaper. Why? *All* the auto manufacturers installed robots, so they've *all* got the same incentive to keep prices the same and pocket the additional productivity as profits. A”.

**Measured discrepancy:** Exact nDCG@10 `0.3155`, HNSW nDCG@10 `0.0000`, nDCG loss `0.3155`, ANN recall@10 `0.00`.

**Assistant draft analysis:** The judged-relevant passage addresses “If early exercise is a bad idea, why American option is more expensive than European [duplicate]”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 31 — query 6668

**Question:** Approximate IT company valuation (to proximate stock options value)

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `275902` (relevance `1`) says: “This situation sounds better than most, the company it seems likely to be profitable in the future. As such it is a good candidate to have a successful IPO. With that your stock options are likely to be worth something. How much of that is your share is likely to be very small. T”.

**Exact top-1:** Document `65900` (relevance `0`) — “"With options you pay for a premium which relates to the expected (so-called ""implied"" volatility). With futures, there is no assumption about the volatility of an underlying stock. In general, when trading options you trade the direction and future expected volatility of an un”.

**HNSW ef=10 top-1:** Document `65900` (relevance `0`) — “"With options you pay for a premium which relates to the expected (so-called ""implied"" volatility). With futures, there is no assumption about the volatility of an underlying stock. In general, when trading options you trade the direction and future expected volatility of an un”.

**Measured discrepancy:** Exact nDCG@10 `0.3066`, HNSW nDCG@10 `0.0000`, nDCG loss `0.3066`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “Approximate IT company valuation (to proximate stock options value)”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 32 — query 1284

**Question:** Tax consequences when foreign currency changes in value

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `342411` (relevance `1`) says: “If you buy foreign currency as an investment, then the gains are ordinary income. The gains are realized when you close the position, and whether you buy something else go back to the original form of investment is of no consequence. In case #1 you have $125 income. In case #2 yo”.

**Exact top-1:** Document `149962` (relevance `0`) — “The closest thing that you are looking for would be FOREX exchanges. Currency value is affected by the relative growth of economies among other things, and the arbritrage of currencies would enable you to speculate on the relative growth of an individual economy.”.

**HNSW ef=10 top-1:** Document `149962` (relevance `0`) — “The closest thing that you are looking for would be FOREX exchanges. Currency value is affected by the relative growth of economies among other things, and the arbritrage of currencies would enable you to speculate on the relative growth of an individual economy.”.

**Measured discrepancy:** Exact nDCG@10 `0.3010`, HNSW nDCG@10 `0.0000`, nDCG loss `0.3010`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “Tax consequences when foreign currency changes in value”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 33 — query 4920

**Question:** Does financing a portfolio on margin affect the variance of a portfolio?

**Suggested category (requires human confirmation):** 2 — Lower ranking utility than exact retrieval (some relevance survives)

**Relevant evidence:** Judged-relevant document `228341` (relevance `1`) says: “"Financing a portfolio with debt (on margin) leads to higher variance. That's the WHOLE POINT. Let's say it's 50-50. On the downside, with 100% equity, you can never lose more than your whole equity. But if you have assets of 100, of which 50% is equity and 50% is debt, your loss”.

**Exact top-1:** Document `228341` (relevance `1`) — “"Financing a portfolio with debt (on margin) leads to higher variance. That's the WHOLE POINT. Let's say it's 50-50. On the downside, with 100% equity, you can never lose more than your whole equity. But if you have assets of 100, of which 50% is equity and 50% is debt, your loss”.

**HNSW ef=10 top-1:** Document `306815` (relevance `1`) — “Yes, more leverage increases the variance of your individual portfolio (variance of your personal net worth). The simple way to think about it is that if you only own only 50% of your risky assets, then you can own twice as many risky assets. That means they will move around twic”.

**Measured discrepancy:** Exact nDCG@10 `0.7654`, HNSW nDCG@10 `0.4693`, nDCG loss `0.2961`, ANN recall@10 `0.70`.

**Assistant draft analysis:** The judged-relevant passage addresses “Does financing a portfolio on margin affect the variance of a portfolio?”, whereas the HNSW top-ranked passage begins with the different material quoted above. The ef=10 ranking retained some relevance but lost nDCG relative to the exact nearest-neighbor ranking. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 34 — query 604

**Question:** Is there a dollar amount that, when adding Massachusetts Sales Tax, precisely equals $200?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `231947` (relevance `1`) says: “"Don't worry about it. The State doesn't care about rounding error. All you need to do is say ""We charge our prices with tax included"" - you know, like carnivals and movie theaters. Then follow the procedures your state specifies for computing reportable tax. Quite likely it wa”.

**Exact top-1:** Document `436128` (relevance `0`) — “You should total the items first, to get $3.00, then add the tax, then round up/down accordingly. Your two examples above don't offer this option, even though your second example arrives at the same result. In your first example, a number of items taxed one at a time might result”.

**HNSW ef=10 top-1:** Document `436128` (relevance `0`) — “You should total the items first, to get $3.00, then add the tax, then round up/down accordingly. Your two examples above don't offer this option, even though your second example arrives at the same result. In your first example, a number of items taxed one at a time might result”.

**Measured discrepancy:** Exact nDCG@10 `0.2961`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2961`, ANN recall@10 `0.70`.

**Assistant draft analysis:** The judged-relevant passage addresses “Is there a dollar amount that, when adding Massachusetts Sales Tax, precisely equals $200?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 35 — query 7823

**Question:** Retirement Funds: Betterment vs Vanguard Life strategy vs Target Retirement

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `105666` (relevance `1`) says: “"First, congratulations on choosing to invest in low cost passively managed plans. If you choose any one of these options and stick with it, you will already be well ahead of most individual investors. Almost all plans will allow you to re-balance between asset classes. With some”.

**Exact top-1:** Document `451170` (relevance `0`) — “Your long-term saving targets will include retirement, kids' college, house, etc. Medium-term might be your college, or a car. Short-term might be a vacation somewhere or a new laptop. In all cases saving, then spending money you do have is better than spending money you don't ha”.

**HNSW ef=10 top-1:** Document `148541` (relevance `0`) — “"Your only real alternative is something like T-Bills via your broker or TreasuryDirect or short-term bond funds like the Vanguard Short-Term Investment-Grade Fund. The problem with this strategy is that these options are different animals than a money market. You're either going”.

**Measured discrepancy:** Exact nDCG@10 `0.2961`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2961`, ANN recall@10 `0.70`.

**Assistant draft analysis:** The judged-relevant passage addresses “Retirement Funds: Betterment vs Vanguard Life strategy vs Target Retirement”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 36 — query 10137

**Question:** F-1 student investing in foreign markets

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `57168` (relevance `1`) says: “You cannot have off-campus employment in your first year, but investments are considered passive income no matter how much time you put into that effort. Obviously you need to stay enrolled full-time and get good enough grades to stay in good standing academically, so you should”.

**Exact top-1:** Document `78632` (relevance `0`) — “As an F1 student, I have been investing (and occasionally buying and selling within few weeks) for several years, and I have never had problems (of course I report to IRS gains/losses every year at tax time). On the other hand, the officer in charge of foreign students at my scho”.

**HNSW ef=10 top-1:** Document `571913` (relevance `0`) — “I am a firm believer in TD's e-series funds. No other bank in Canada has index funds with such low management fees. Index funds offer the flexibility to re-balance your portfolio every month without the need to pay commission fees. Currently I allocate 10% of my paycheck to be di”.

**Measured discrepancy:** Exact nDCG@10 `0.2891`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2891`, ANN recall@10 `0.40`.

**Assistant draft analysis:** The judged-relevant passage addresses “F-1 student investing in foreign markets”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 37 — query 515

**Question:** financial institution wants share member break down for single member LLC

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `372909` (relevance `1`) says: “"What exactly would the financial institution need to see to make them comfortable with these regulations The LLC Operating Agreement. The OA should specify the member's allocation of equity, assets, income and loss, and of course - managerial powers and signature authorities. In”.

**Exact top-1:** Document `334603` (relevance `0`) — “"If you have a single member LLC there is no need to separate expenses in this way since it is simply treated as part of the owner's normal tax returns. This is the way I've been operating. Owner of Single-Member LLC If a single-member LLC does not elect to be treated as a corpor”.

**HNSW ef=10 top-1:** Document `334603` (relevance `0`) — “"If you have a single member LLC there is no need to separate expenses in this way since it is simply treated as part of the owner's normal tax returns. This is the way I've been operating. Owner of Single-Member LLC If a single-member LLC does not elect to be treated as a corpor”.

**Measured discrepancy:** Exact nDCG@10 `0.2891`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2891`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “financial institution wants share member break down for single member LLC”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 38 — query 2330

**Question:** How can I determine if a debt consolidation offer is real or a scam?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `104221` (relevance `1`) says: “I think in such situations a good rule of thumb may be - if you are asked to pay significant sums of money upfront before anything is done, stop and ask yourself, what would you do if they don't do what they promised? They know who you are, but usually most you know is a company”.

**Exact top-1:** Document `179808` (relevance `0`) — “Debt consolidation is basically getting all your debt into one loan. This is possibly more convenient, and lets you close the other accounts (in the case of credit cards, preventing you from incurring any more debt). Ideally, your consolidated debt will have a better interest rat”.

**HNSW ef=10 top-1:** Document `179808` (relevance `0`) — “Debt consolidation is basically getting all your debt into one loan. This is possibly more convenient, and lets you close the other accounts (in the case of credit cards, preventing you from incurring any more debt). Ideally, your consolidated debt will have a better interest rat”.

**Measured discrepancy:** Exact nDCG@10 `0.2372`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2372`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “How can I determine if a debt consolidation offer is real or a scam?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 39 — query 6219

**Question:** Are there Investable Real Estate Indices which track Geographical Locations?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `146924` (relevance `1`) says: “Not to my knowledge. Often the specific location is diversified out of the fund because each major building company or real estate company attempts to diversify risk by spreading it over multiple geographical locations. Also, buyers of these smaller portfolios will again diversif”.

**Exact top-1:** Document `452122` (relevance `0`) — “More importantly, index funds are denominated in specific currencies. You can't buy or sell an index, so it can be dimensionless. Anything you actually do to track the index involves real amounts of real money.”.

**HNSW ef=10 top-1:** Document `103959` (relevance `0`) — “"An index annuity is almost the same as Indexed Universal Life, except the equity-index annuity is an investment with a guaranteed minimum return, with sometimes a higher return that is a function of the gain in the stock market, but is not associated with a life insurance policy”.

**Measured discrepancy:** Exact nDCG@10 `0.2372`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2372`, ANN recall@10 `0.30`.

**Assistant draft analysis:** The judged-relevant passage addresses “Are there Investable Real Estate Indices which track Geographical Locations?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 40 — query 4011

**Question:** How can I deal with a spouse who compulsively spends?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `136367` (relevance `1`) says: “"compulsive eating, and other compulsions, are also an issue If this is true, then this is not a money problem. This is a psychological problem that manifests itself in overspending. I would make an appointment with a counselor or therapist ASAP to start dealing with this problem”.

**Exact top-1:** Document `126455` (relevance `0`) — “Other than the two answers (both of which recommend waiting until marriage to actually combine finances, and which I agree with), there's the general question: how does a couple choose to manage finances? In our marriage, it's me. I'm more numbers-minded than my spousal-unit. I'm”.

**HNSW ef=10 top-1:** Document `126455` (relevance `0`) — “Other than the two answers (both of which recommend waiting until marriage to actually combine finances, and which I agree with), there's the general question: how does a couple choose to manage finances? In our marriage, it's me. I'm more numbers-minded than my spousal-unit. I'm”.

**Measured discrepancy:** Exact nDCG@10 `0.2346`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2346`, ANN recall@10 `0.70`.

**Assistant draft analysis:** The judged-relevant passage addresses “How can I deal with a spouse who compulsively spends?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 41 — query 8002

**Question:** What is the tax treatment of scrip dividends in the UK?

**Suggested category (requires human confirmation):** 2 — Lower ranking utility than exact retrieval (some relevance survives)

**Relevant evidence:** Judged-relevant document `118786` (relevance `1`) says: “I wrote about this in another answer: You can sell the scrip dividend in the market; the capital gain from this sale may fall below the annual tax-free allowance for capital gains, in which case you don't pay any capital gains tax on that amount. For a cash dividend, however, the”.

**Exact top-1:** Document `217006` (relevance `0`) — “Am I correct in understanding that a Scrip Dividend involves the issue of new shares instead of the purchase of existing shares? Yes. Instead of paying a cash dividend to shareholders, the company grants existing shareholders new shares at a previously determined price. This allo”.

**HNSW ef=10 top-1:** Document `97842` (relevance `0`) — “Income and Capital are taxed separately in the uk. You probably can't get dividends paid gross even in ISA's you pay the basic rate of tax on dividends only higher rate tax payers get tax benefit from dividends. What you could do is invest in splits (Spilt capital investment trus”.

**Measured discrepancy:** Exact nDCG@10 `0.4525`, HNSW nDCG@10 `0.2346`, nDCG loss `0.2179`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “What is the tax treatment of scrip dividends in the UK?”, whereas the HNSW top-ranked passage begins with the different material quoted above. The ef=10 ranking retained some relevance but lost nDCG relative to the exact nearest-neighbor ranking. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 42 — query 1736

**Question:** How can people have such high credit card debts?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `25543` (relevance `1`) says: “I had $70K in credit card at one point. Limited income, starting a business - it's the only credit available. (yes, all paid off now).”.

**Exact top-1:** Document `188406` (relevance `0`) — “Credit scoring has changed since the time of this question (July 2017) and it is now possible that having a high available credit balance can negatively affect your credit score. ... VantageScore will now mark a borrower negatively for having excessively large credit card limits,”.

**HNSW ef=10 top-1:** Document `188406` (relevance `0`) — “Credit scoring has changed since the time of this question (July 2017) and it is now possible that having a high available credit balance can negatively affect your credit score. ... VantageScore will now mark a borrower negatively for having excessively large credit card limits,”.

**Measured discrepancy:** Exact nDCG@10 `0.2140`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2140`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “How can people have such high credit card debts?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 43 — query 5241

**Question:** Mortgage vs. Cash for U.S. home buy now

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `234286` (relevance `1`) says: “"If you are investing in a mortgage strictly to avoid taxes, the answer is ""pay cash now."" A mortgage buys you flexibility, but at the cost of long term security, and in most cases, an overall decrease in wealth too. At a very basic level, I have to ask anyone why they would pa”.

**Exact top-1:** Document `273735` (relevance `0`) — “First, who is saying that it is a better option? In general it is best to pay cash for things when you can. I think the reality is that for most people owning a house would be very difficult without some sort of financing. That said, one argument for financing a house these days”.

**HNSW ef=10 top-1:** Document `273735` (relevance `0`) — “First, who is saying that it is a better option? In general it is best to pay cash for things when you can. I think the reality is that for most people owning a house would be very difficult without some sort of financing. That said, one argument for financing a house these days”.

**Measured discrepancy:** Exact nDCG@10 `0.2140`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2140`, ANN recall@10 `0.70`.

**Assistant draft analysis:** The judged-relevant passage addresses “Mortgage vs. Cash for U.S. home buy now”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 44 — query 3149

**Question:** Tips for insurance coverage for one-man-teams

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `490223` (relevance `1`) says: “While the OP disses the health insurance coverage offered through his wife's employer as a complete rip-off, one advantage of such coverage is that, if set up right (by the employer), the premiums can be paid for through pre-tax dollars instead of post-tax dollars. On the other h”.

**Exact top-1:** Document `456456` (relevance `0`) — “&gt;I want a president who is going to look out for the little guys. Obama certainly wouldn't be your man then. Consider his insurance, and wall street ties; he's not for the little guy; he may say he is, but his actions and policy are obviously not for them. Also, consider that”.

**HNSW ef=10 top-1:** Document `456456` (relevance `0`) — “&gt;I want a president who is going to look out for the little guys. Obama certainly wouldn't be your man then. Consider his insurance, and wall street ties; he's not for the little guy; he may say he is, but his actions and policy are obviously not for them. Also, consider that”.

**Measured discrepancy:** Exact nDCG@10 `0.2044`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2044`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “Tips for insurance coverage for one-man-teams”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 45 — query 2551

**Question:** How to find cheaper alternatives to a traditional home telephone line?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `143100` (relevance `1`) says: “Try to use VOIP service provider or web enabled conference calling services in your home phone. Now a days communication technologies have seen a boost as well as integration of different formats and platforms which easily reduces phone bills of a user. Service such as UberConfer”.

**Exact top-1:** Document `482638` (relevance `0`) — “For alternative financing, pursue a line of credit or a Home Equity Line of Credit. (From the comments of @ChrisInEdmonton and @littleadv on the original question)”.

**HNSW ef=10 top-1:** Document `555963` (relevance `0`) — “Buy best phones within affordable prices, now you can buy the Samsung, android smart phones online through at shopallitems,which is one of the best online stores in the Dubai, UAE. We are offering a Samsung J1 mini and new samsung phone J1 prime which available in the AED350. 00”.

**Measured discrepancy:** Exact nDCG@10 `0.2021`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2021`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “How to find cheaper alternatives to a traditional home telephone line?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 46 — query 5271

**Question:** Why are auto leases stubbornly strict about visa status and how to work around that?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `221439` (relevance `1`) says: “When getting a car always start with your bank or credit union. They are very likely to offer better loan rate than the dealer. Because you start there you have a data point so you can tell if the dealer is giving you a good rate. Having the loan approved before going to the deal”.

**Exact top-1:** Document `598414` (relevance `0`) — “"Verbal agreements are not legally binding. Unless you have signed a new lease agreement, you are not obligated to continue renting the property - you are free to go. On the other hand, if you really like the place and want to stay, you should sign another lease agreement. This a”.

**HNSW ef=10 top-1:** Document `598414` (relevance `0`) — “"Verbal agreements are not legally binding. Unless you have signed a new lease agreement, you are not obligated to continue renting the property - you are free to go. On the other hand, if you really like the place and want to stay, you should sign another lease agreement. This a”.

**Measured discrepancy:** Exact nDCG@10 `0.2021`, HNSW nDCG@10 `0.0000`, nDCG loss `0.2021`, ANN recall@10 `0.70`.

**Assistant draft analysis:** The judged-relevant passage addresses “Why are auto leases stubbornly strict about visa status and how to work around that?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 47 — query 9646

**Question:** Do common stocks and preferred stocks have any differences in terms of percentage of the company per unit they represent?

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `272784` (relevance `1`) says: “Preferred stocks are, err... Preferred. The whole point of preferred stocks is that they have some preference over other classes of stocks (there may be more than 2, by the way). It can be more voting rights, more dividends or priority on dividends' distribution (common with VC i”.

**Exact top-1:** Document `503740` (relevance `0`) — “Stocks represent partial ownership of the company. So, if you owned 51% of the stock of the company (and therefore 51% of the company itself), you could decide to liquidate all the assets of the company, and you would be entitled to 51% of the proceeds from that sale. In the exam”.

**HNSW ef=10 top-1:** Document `50081` (relevance `0`) — “What percent of a company are you buying when you purchase stock? The percent of a company represented by a single share can be calculated by percent = 1/number_of_shares*100% Apple comprises 5,250,000,000 shares, so one share makes up about 1.9e-8% of a company, or 0.000000019%”.

**Measured discrepancy:** Exact nDCG@10 `0.1934`, HNSW nDCG@10 `0.0000`, nDCG loss `0.1934`, ANN recall@10 `0.40`.

**Assistant draft analysis:** The judged-relevant passage addresses “Do common stocks and preferred stocks have any differences in terms of percentage of the company per unit they represent?”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 48 — query 5064

**Question:** How much should a graduate student attempt to save?

**Suggested category (requires human confirmation):** 2 — Lower ranking utility than exact retrieval (some relevance survives)

**Relevant evidence:** Judged-relevant document `192811` (relevance `1`) says: “"First, don't save anything in a tax sheltered vehicle. You will be paying so little tax that there will be essentially no benefit to making the contributions, and you'll pay tax when they come out. Tax free compounding for 40 years is terrific, but start that after you're earnin”.

**Exact top-1:** Document `440628` (relevance `0`) — “To make this more general: should you save for retirement or for education? The answer is simple: save for your education first. A PriceWaterhouseCoopers report on behalf Universities UK in 2006 and another by the Washington-based College Board in 2007 indicate that graduates of”.

**HNSW ef=10 top-1:** Document `440628` (relevance `0`) — “To make this more general: should you save for retirement or for education? The answer is simple: save for your education first. A PriceWaterhouseCoopers report on behalf Universities UK in 2006 and another by the Washington-based College Board in 2007 indicate that graduates of”.

**Measured discrepancy:** Exact nDCG@10 `0.3978`, HNSW nDCG@10 `0.2184`, nDCG loss `0.1794`, ANN recall@10 `0.80`.

**Assistant draft analysis:** The judged-relevant passage addresses “How much should a graduate student attempt to save?”, whereas the HNSW top-ranked passage begins with the different material quoted above. The ef=10 ranking retained some relevance but lost nDCG relative to the exact nearest-neighbor ranking. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 49 — query 6554

**Question:** Mutual fund value went down, shares went up, no action taken by me

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `22469` (relevance `1`) says: “It is very likely that the fund paid out a dividend in the form of reinvested shares. This happens with many funds, especially as we come to the end of the year. Here's a simplified example of how it works. Assume you invested $1000 and bought 100 units at $10/unit. Ignoring the”.

**Exact top-1:** Document `477092` (relevance `0`) — “The volatility of an index fund should usually be a lot lower than that of an individual stock. However even with a broad index fund you should consider the fact that being down by 10% in the time frame you refer to is quite possible! So is being up by 10% of course. A corporate”.

**HNSW ef=10 top-1:** Document `457811` (relevance `0`) — “Mutual funds don't have intraday prices. They have net asset values which are calculated periodically (daily or weekly or any other period depending on the fund).”.

**Measured discrepancy:** Exact nDCG@10 `0.1772`, HNSW nDCG@10 `0.0000`, nDCG loss `0.1772`, ANN recall@10 `0.50`.

**Assistant draft analysis:** The judged-relevant passage addresses “Mutual fund value went down, shares went up, no action taken by me”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

## Case 50 — query 9060

**Question:** Buying puts without owning underlying

**Suggested category (requires human confirmation):** 1 — Relevant passage missing from approximate top 10

**Relevant evidence:** Judged-relevant document `40447` (relevance `1`) says: “In the money puts and calls are subject to automatic execution at expiration. Each broker has its own rules and process for this. For example, I am long a put. The strike is $100. The stock trades at the close, that final friday for $90. I am out to lunch that day. Figuratively,”.

**Exact top-1:** Document `107688` (relevance `0`) — “Since you mention the religion restriction, you should probably look into the stock market or funds investing in it. Owning stock basically means you own a part of a company and benefit from any increase in value the company may have (and 'loose' on decreases, provided you sell y”.

**HNSW ef=10 top-1:** Document `107688` (relevance `0`) — “Since you mention the religion restriction, you should probably look into the stock market or funds investing in it. Owning stock basically means you own a part of a company and benefit from any increase in value the company may have (and 'loose' on decreases, provided you sell y”.

**Measured discrepancy:** Exact nDCG@10 `0.1772`, HNSW nDCG@10 `0.0000`, nDCG loss `0.1772`, ANN recall@10 `0.70`.

**Assistant draft analysis:** The judged-relevant passage addresses “Buying puts without owning underlying”, whereas the HNSW top-ranked passage begins with the different material quoted above. Under the dataset's relevance judgments, ef=10 returned no judged-relevant result in its top 10. The small graph-search budget is a plausible contributing factor, but the exact cause is not established by these metrics alone.

**Suggested investigation:** Consult the saved `fiqa_query_budget_results.csv` for this exact query at larger `efSearch` values; record the smallest observed budget that recovers acceptable nDCG and compare the passages. Consider whether ambiguous terminology or missing representation cues are contributing, but **do not present untested mechanisms as proven**.

**Reviewer to verify:** Read the complete judged-relevant, exact top-5, and HNSW top-5 passages in the original worksheet. Correct or replace the suggested category as appropriate, then enter your own evidence-backed explanation, source doc IDs/ranks and remedy. Only then mark complete.

---
**Automatically summarized outcome groups:** 44 cases with HNSW nDCG@10=0; 6 cases with degraded but nonzero HNSW nDCG@10. These are **not validated human failure taxonomy counts**.
