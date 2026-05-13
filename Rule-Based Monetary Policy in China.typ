// Rule-Based Monetary Policy in China: An Empirics-Only Counterfactual Analysis
// Converted from Lin_Research Proposal.tex

#set document(
  title: "Rule-Based Monetary Policy in China: An Empirics-Only Counterfactual Analysis",
  author: "Liming Lin",
)

#set page(
  paper: "a4",
  margin: 1in,
)

#set text(
  font: "New Computer Modern",
  size: 12pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.85em, // approximates one-half spacing
  first-line-indent: 1.5em,
)

#set heading(numbering: "1.1")

// Title page
#align(center)[
  #v(1em)
  #text(size: 16pt, weight: "bold")[
    Rule-Based Monetary Policy in China: \
    An Empirics-Only Counterfactual Analysis
  ]

  #v(1em)
  #text(size: 12pt)[
    Liming Lin#footnote[Sciences Po Paris. Supervisor: Paul Bouscasse. Jury: Paul Bouscasse, Axelle Ferrière. I am grateful to ... All errors are my own.]
  ]

  #v(0.5em)
  May 2026
]

#v(2em)

// Abstract
#align(center)[
  #text(weight: "bold")[Abstract]
]

#block(inset: (left: 0.5in, right: 0.5in))[
  #set par(first-line-indent: 0em)
  Since 2023, Chinese CPI inflation has hovered near zero, persistently undershooting the targets set in the annual Government Work Report. This paper applies the empirics-only counterfactual framework of Caravello, McKay and Wolf (2025) to ask what would have happened to Chinese inflation, output, and the exchange rate if the People's Bank of China had instead followed an explicit targeting rule over this period. Two sets of impulse response functions to monetary policy shocks --- one identified from a narrative policy-rule residual on the 7-day repo rate (FR007), the other from high-frequency FR007 surprises on policy event dates --- are combined with a 7-variable Bayesian VAR baseline projection in a stacked least-squares solver to recover the shock sequence that best implements alternative targeting rules. The headline finding is a persistent gap between counterfactual paths and the Government Work Report CPI targets, even under aggressive strict targeting, suggesting that conventional interest rate policy alone could not have closed the realized inflation gap. The result is robust to flexible targeting that includes output and exchange-rate stabilization objectives, and to extending the estimation sample to 2025. A secondary finding --- a positive short-run CPI response to a contractionary FR007 shock that appears in both identification schemes --- is interpreted as a substantive feature of China's CPI methodology, in which floating-rate mortgage interest enters the housing sub-index, rather than as a model deficiency.

  #v(0.5em)
  *Keywords:* monetary policy, counterfactuals, China, inflation targeting, sufficient statistics. \
  *JEL codes:* E52, E58, E31.
]

#v(2em)

= Introduction

Since 2023, Chinese CPI inflation has hovered near zero, persistently undershooting the official targets set in the annual Government Work Report (GWR). The GWR CPI targets for 2023, 2024, and 2025 were 3%, 3%, and roughly 2%, respectively; realized year-on-year CPI growth over the same period averaged close to zero. Producer prices have remained in negative territory since late 2022, and the deflationary stance has coincided with severe liquidity stress in the real estate sector --- exemplified by the bankruptcies and near-defaults of major developers such as Evergrande and Country Garden --- and with elevated local government financing-vehicle (LGFV) debt risks.

Despite these pressures, the People's Bank of China (PBoC) has maintained a cautious, gradualist stance. The Loan Prime Rate (LPR), now the principal anchor for new loan pricing, has been lowered only moderately, from 3.65% in 2022 to roughly 3.0% in early 2026, a pace that has been widely viewed as insufficient to arrest the deflationary spiral.#footnote[See, for example, the discussion of monetary policy options in the IMF's Article IV consultations with China for 2024 and 2025.] The 2024 Central Economic Work Conference (CEWC) introduced for the first time the formulation of "a reasonable rebound in the price level" (_wujia heli huisheng_, 物价合理回升); the 2025 CEWC explicitly tied this objective to the conduct of monetary policy. These developments raise a fundamental question about the recent past: was the discretionary failure to meet the inflation target a result of structural constraints on monetary policy in China, or was it primarily a policy choice? Equivalently, what would the trajectory of Chinese CPI, output, and the exchange rate have looked like over 2023--2025 if the PBoC had instead followed an explicit, rule-based targeting policy?

*Approach.* This paper answers that question by applying the empirics-only counterfactual framework of #cite(<wolf2025>, form: "prose") to China.#footnote[Throughout the paper I refer to @wolf2025 as "CMW2025" and to the foundational result in @mckay2023 as "MW2023".] The approach builds on the sufficient-statistics insight of #cite(<mckay2023>, form: "prose") and #cite(<barnichon2023>, form: "prose"): under weak structural assumptions, evaluating macroeconomic outcomes under alternative monetary policy rules requires only two empirically estimable objects --- a reduced-form baseline projection of the economy and the causal impulse-response functions (IRFs) of macro variables to monetary policy shocks. CMW2025 show that, when the contemplated counterfactual involves changes to the short end of the yield curve, these objects can be taken directly from the data, sidestepping the need for a fully specified structural model. The framework is therefore well-suited to settings where the underlying structure is poorly known --- arguably the case for China, where multi-instrument policy, dual-track interest rates, and tight monetary--fiscal coordination complicate any DSGE-style identification of mechanisms.

The empirical implementation proceeds in four blocks, in the spirit of the CMW2025 application to U.S. post-COVID inflation. First, a 7-variable Bayesian VAR with Minnesota prior, estimated on monthly data from 2002 onward, produces a baseline projection of the Chinese economy from a chosen forecast date. Second, narrative monetary policy shocks are identified as the residual of a forward-looking policy rule for FR007 that combines elements from #cite(<chen2025>, form: "prose") (asymmetric output gap, lagged interest rate) and #cite(<rey2026>, form: "prose") (post-2006 exchange-rate gap term). Third, high-frequency shocks are constructed as daily close-to-close FR007 changes on dates of PBoC announcements that involve a change in one of the core policy instruments --- 7-day reverse repo, 1-year MLF, 1-year LPR, RRR for large financial institutions, and pre-2015 benchmark deposit/lending rates --- and aggregated to monthly frequency. Fourth, the shock sequence that minimizes a weighted quadratic loss over CPI deviations from the GWR target, output deviations from the GWR growth target, FR007 first differences (a smoothness penalty), and policy-induced deviations of the nominal effective exchange rate from baseline is recovered from a stacked-OLS problem in which both narrative and HFI transmission maps enter as columns of the policy-shock causal-effect matrix.

*Findings.* Three results emerge.

First, even under strict CPI targeting --- in which the loss function places weight only on the inflation gap and an interest-rate smoothness penalty --- the counterfactual CPI path remains substantially below the GWR targets throughout 2023--2025. Adding flexible targeting motives (output stabilization, exchange-rate stability) tightens the gap further. The relevant comparison here is between counterfactual CPI and the GWR target, not between counterfactual and baseline: the BVAR baseline is itself notably pessimistic from 2023 onward, likely because the model extrapolates post-COVID deflationary dynamics that may represent a structural break from the estimation sample. The persistent gap between the counterfactual and the announced target is the central finding.

Second, the counterfactual reveals a clear policy trade-off. Across all targeting rules considered, more aggressive rate cuts achieve lower (more negative) counterfactual inflation deviations, consistent with a price-puzzle pattern in the underlying IRFs (see below); the cost is larger NEER swings and greater output volatility relative to more balanced rules.

Third, both identification schemes deliver an initial _positive_ CPI response to a contractionary FR007 shock --- a price puzzle. Rather than treating this as a deficiency to be patched away (e.g., by adding commodity-price controls or sign restrictions), I follow #cite(<chodorow2025>, form: "prose") in interpreting it as a substantive feature of Chinese CPI methodology. The Chinese CPI's housing sub-index (居住类) employs a user-cost approach that includes mortgage interest payments; with the bulk of Chinese mortgages on floating rates pegged to the LPR, an upward FR007 shock that propagates to the LPR mechanically raises the housing sub-index in the short run. The puzzle is therefore informative about the housing channel of monetary transmission, not a sign that the shocks are misidentified.

*Contributions.* This paper makes three contributions.

_First_, it is, to my knowledge, the first application of the empirics-only counterfactual framework of #cite(<wolf2025>, form: "prose") to a non-U.S., emerging-market context.#footnote[#cite(<bouscasse2024>, form: "prose") apply the related sufficient-statistics framework of @mckay2023 to fiscal counterfactuals in the U.S. post-COVID inflation episode; the present paper is closer in spirit to that exercise than to the original U.S. monetary applications in CMW2025.] The application is non-trivial: the framework was originally designed around the federal funds rate as a single, well-identified policy instrument, while the PBoC operates a multi-instrument framework with dual-track rates. The paper takes a clear stand on this issue by selecting FR007 --- the market rate that anchors HFI surprises and that #cite(<chen2025>, form: "prose") use as the dependent variable in their policy rule --- as the unifying instrument across all four building blocks.

_Second_, the paper combines two distinct shock-identification schemes within a single counterfactual exercise. Following the CMW2025 logic, both narrative and HFI shocks enter as columns of the transmission map, allowing the solver to choose a mixture rather than treating one as primary and the other as robustness. This is more demanding than the standard practice in the China monetary-policy literature, where a single shock series is typically estimated and compared across alternatives.

_Third_, the paper documents and rationalizes a robust price puzzle in Chinese CPI responses to monetary policy shocks. The interpretation through the housing user-cost channel, building on #cite(<chodorow2025>, form: "prose"), has implications for the design of future inflation-targeting frameworks in China and for the cross-country literature on monetary transmission to housing.

*Roadmap.* @sec:litreview reviews the related literature on monetary policy identification in China, on counterfactual policy evaluation methods, and on the recent Chinese deflationary episode. @sec:institutional provides institutional background on the PBoC's policy framework and on the Government Work Report target system. @sec:framework sets out the empirical framework, including the BVAR baseline projection, the two shock-identification schemes, and the counterfactual solver. @sec:data describes the data. @sec:irf presents the estimated impulse responses and discusses the price puzzle. @sec:counterfactual reports the main counterfactual results. @sec:discussion discusses limitations and avenues for future research. @sec:conclusion concludes.

= Related Literature <sec:litreview>

This paper sits at the intersection of three literatures.

== Monetary policy identification in China

// [TO WRITE: ~1 page, organized as two sub-traditions]
// Sub-tradition 1: narrative / policy-rule residual approaches.
//   - Chen, Ren & Zha (2018): shadow banking, asymmetric rule on M2 growth.
//   - Chen, Xiao & Zha (2025): updated rule on FR007, regime-switching, systemic risk.
//   - Miranda-Agrippino, Nenova & Rey (2026, "MANR2026"): CMPI as the LHS,
//     forward-looking lag terms, exchange-rate gap.
//   - Position: this paper combines the FR007 dependent variable from CXZ2025
//     with the FX gap and forward-looking lag structure from MANR2026.
// Sub-tradition 2: high-frequency identification.
//   - Das & Song (2022) IMF WP: 1-year IRS surprises around policy events;
//     also introduces a text-based monetary-fiscal coordination measure.
//   - He, Jia, Li & Wu (2024): Chinese adaptation of Bu, Rogers & Wu (2021)
//     yield-curve filtering approach.
//   - Bu, Rogers & Wu (2021) for the underlying U.S. method.
//   - Position: this paper uses FR007 (not IRS, due to data access constraints)
//     and restricts to rate-change-only event dates.

== Counterfactual monetary policy evaluation

// [TO WRITE: ~3/4 page]
// Standard structural tradition: Smets-Wouters (2007); Bocola, Dovis,
// Fella & Roldan-Blanco (2024) on U.S. post-COVID; mention Lucas (1976).
// Sufficient-statistics revolution: McKay & Wolf (2023, Econometrica);
// Barnichon & Mesters (2023). Identification via causal effects of policy
// shocks rather than full structural model.
// Hybrid empirics-only: Caravello, McKay & Wolf (2025). Key insight: when
// counterfactual involves only short-end changes, empirical IRFs alone suffice;
// structural extrapolation only needed for the long end of the yield curve.
// Recent applications: Bouscasse & Hu (2024) on fiscal counterfactuals.
// Position: first emerging-market application of the CMW2025 empirics-only
// approach; methodologically aligned with the post-COVID U.S. exercise.

== The price puzzle, housing channels, and Chinese CPI

// [TO WRITE: ~3/4 page]
// Empirical price puzzle in monetary VARs (Sims 1992; Christiano-Eichenbaum-Evans 1999;
// Castelnuovo-Surico 2010 review) --- usually attributed to omitted variables
// (commodity prices) or imperfect identification.
// Chodorow-Reich (2025, "Housing in the CPI"): user-cost vs. OER methodologies
// for CPI housing components; implications for short-run CPI response to
// monetary policy.
// China-specific: NBS uses a user-cost approach for owner-occupied housing
// (居住类 sub-index includes mortgage interest), unlike the U.S. OER method.
// Combined with floating-rate mortgages anchored to the LPR, this generates
// a mechanical short-run positive CPI response to contractionary policy.
// Position: this paper interprets the price puzzle as a feature of CPI
// measurement rather than a sign of misidentification --- aligning with
// Chodorow-Reich's framework.
// (Optional sub-thread for fiscal-monetary coordination / information effects:
//   Das & Song (2022) text-based measure; Bauer-Swanson (2023) on news vs. shocks;
//   Nakamura-Steinsson (2018), Jarocinski-Karadi (2020) on information effects;
//   Barthelemy-Mengus-Plantin (2024) on fiscal dominance --- mentioned briefly
//   as motivation for future work in §8.)

= Institutional Background <sec:institutional>

// [TO WRITE: ~1-1.5 pages]
// 3.1 The PBoC's policy framework: multi-instrument, gradual transition from
//     quantity to price targeting; FR007 as the de facto market anchor since
//     ~2016; LPR reform of August 2019; abandonment of the 1-year MLF in 2024.
// 3.2 The Government Work Report target system: NPC announcement timing
//     (March), CPI and growth targets, comparison with Western inflation-targeting
//     frameworks (no formal IT mandate, but explicit numerical targets that
//     create de facto accountability).
// 3.3 The 2023-2025 deflationary episode: timeline, key events, contrast with
//     prior PBoC behavior (e.g., 2008-09, 2015-16, 2020).

= Empirical Framework <sec:framework>

// This is where most of the methodology content from the existing proposal
// and from the slides goes. Subsections:

== Sufficient statistics for counterfactual policy evaluation

// [Adapted from CMW2025 introduction. Statement of Proposition 1 (Wold
// representation + invertibility), and the four-block decomposition.]

== Baseline projection: Bayesian VAR

// [Slides Block 3 content. 7-variable monthly BVAR with p=6 lags,
// Minnesota prior, COVID dummies. Variable list and ordering.
// Estimation sample decisions.]

== Narrative shocks: a forward-looking policy rule for FR007

// [Slides Block 1 content + content from existing proposal §3.6.
// Specification of the rule; motivation for hybrid CXZ2025 + MANR2026
// specification; asymmetric output gap; FX gap construction.]

== High-frequency shocks: FR007 surprises on PBoC event dates

// [Slides Block 2 content. Event date construction from CMPI.xlsx;
// the 5 core policy instruments tracked; rate-change-only restriction
// and rationale; daily-to-monthly aggregation by summation.]

== The counterfactual solver

// [Slides Block 4 content. Loss function; stacked-OLS solution;
// the two-treatment Toeplitz construction (Pi_m as T x 2);
// Wolf-style first-period wedge for the discontinuous FR007 jump.]

= Data <sec:data>

// [TO WRITE: ~1 page]
// Monthly GDP construction following Stock & Watson (2010) using consumption,
// FAI, government spending, trade balance.
// Inflation, IP, M2, FR007, NEER, US IP --- sources, frequency, sample.
// Event date construction from CMPI.xlsx.
// Government Work Report targets (3/3/3/3/2 for 2021--2025) --- step in
// March (NPC announcement), not January.

= Estimated Causal Effects <sec:irf>

// [TO WRITE: ~2-3 pages with figures]
// 6.1 The estimated policy rule (Figure: actual vs predicted FR007, residuals).
// 6.2 Narrative IRFs (Figure with credible bands).
// 6.3 HFI IRFs and comparison "any announcement" vs. "rate-change-only".
// 6.4 The price puzzle: discussion through Chodorow-Reich housing channel.
// 6.5 Subsample IRF comparison (2002-2015, 2015-2020, 2020-2025) as
//     evidence of regime change in transmission.

= Counterfactual Analysis <sec:counterfactual>

// [TO WRITE: ~3-4 pages, this is the core results section]
// 7.1 Strict CPI targeting (lambda_pi = lambda_i = 1).
// 7.2 Flexible IT with output (adds lambda_y).
// 7.3 NEER-augmented (adds lambda_e).
// 7.4 Robustness: 2022 vs. 2025 estimation cutoff; 2023:01 vs 2022:12 start.
// 7.5 Discussion: counterfactual vs. GWR target as the main metric;
//     pessimistic baseline as a caveat.

= Discussion and Limitations <sec:discussion>

// [TO WRITE: ~1.5 pages]
// Linearity in monthly shock aggregation (saved research idea 2).
// Single-instrument restriction (multi-instrument PBoC vs. FR007 anchor).
// Information effects in coordinated regimes (idea 4 --- Plantin signal).
// Mid-month clustering / HFI contamination (idea 1).
// Pessimistic baseline as structural-break concern.

= Conclusion <sec:conclusion>

// [TO WRITE: ~1/2 page]

#pagebreak()

// ============================================================================
// References
// ============================================================================

#heading(numbering: none)[References]

#set par(first-line-indent: 0em, hanging-indent: 0.5in)

Barnichon, R., and G. Mesters. (2023). A Sufficient Statistics Approach for Macro Policy Evaluation. _Working Paper_.

Bouscasse, P., and S. Hu. (2024). Fiscal Policy and Monetary Counterfactuals: Evidence from the U.S. Post-COVID Inflation. _Working Paper_.

Bu, Chunya, John Rogers, and Wenbin Wu. (2021). A Unified Measure of Fed Monetary Policy Shocks. _Journal of Monetary Economics_, 118 (March), 331--349.

Caravello, T. E., A. McKay, and C. K. Wolf. (2025). Evaluating Monetary Policy Counterfactuals: (When) Do We Need Structural Models? _Working Paper_.

Chen, K., P. Higgins, and T. Zha. (2024). Constructing Quarterly Chinese Time Series Usable for Macroeconomic Analysis. _Journal of International Money and Finance_, 143, 103052.

Chen, K., J. Ren, and T. Zha. (2018). The Nexus of Monetary Policy and Shadow Banking in China. _American Economic Review_, 108(12), 3891--3936.

Chen, K., Y. Xiao, and T. Zha. (2025). A Trade-off Between Monetary Policy Transmission and Systemic Risk in China. _NBER Working Paper 34056_.

Chodorow-Reich, G. (2025). Housing in the CPI. _Harvard University Working Paper_.

Das, S., and W. Song. (2022). Monetary Policy Transmission and Policy Coordination in China. _IMF Working Paper WP/22/74_.

Fernald, J., E. Hsu, and M. M. Spiegel. (2021). Is China fudging its GDP figures? Evidence from trading partner data. _Journal of International Money and Finance_, 110, 102262.

Giannone, D., M. Lenza, and G. E. Primiceri. (2015). Prior Selection for Vector Autoregressions. _Review of Economics and Statistics_, 97(2), 436--451.

He, J., D. Jia, K. Li, and W. Wu. (2024). A High-Frequency Measure of Chinese Monetary Policy Shocks. _PHBS Working Paper_.

McKay, A., and C. K. Wolf. (2023). What Can Time-Series Regressions Tell Us About Policy Counterfactuals? _Econometrica_, 91(5), 1695--1725.

Miranda-Agrippino, S., T. Nenova, and H. Rey. (2026). A Composite Monetary Policy Indicator for China. _Working Paper_.

Romer, Christina D., and David H. Romer. (2004). A New Measure of Monetary Shocks: Derivation and Implications. _American Economic Review_, 94(4), 1055--1084.

Stock, J. H., and M. W. Watson. (2010). Distribution of Quarterly Values of GDP/GDI Across Months Within the Quarter. _Mimeo_.
