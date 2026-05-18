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
#set page(
  footer: context [
    #align(center)[
      #counter(page).display()
    ]
  ]
)
#show heading: set block(

  above: 1.5em,

  below: 1em,

)

#set heading(numbering: "1.1")

// Formal cover page
#set page(footer: none)
#align(center)[
  #v(3em)
  #text(size: 18pt, weight: "bold")[
    Rule-Based Monetary Policy in China: \
    An Empirics-Only Counterfactual Analysis
  ]

  #v(4em)
  #text(size: 14pt, weight: "bold")[Liming Lin]

  #v(4em)
  #text(weight: "bold")[Supervisor]
  #v(0.5em)
  Paul Bouscasse

  #v(2em)
  #text(weight: "bold")[Jury Members]
  #v(0.5em)
  Paul Bouscasse \
  Axelle Ferrière

  #v(4em)
  May 2026
]

#pagebreak()
#counter(page).update(1)
#set page(
  footer: context [
    #align(center)[
      #counter(page).display()
    ]
  ]
)

// Title page
#align(center)[
  #v(1em)
  #text(size: 16pt, weight: "bold")[
    Rule-Based Monetary Policy in China: \
    An Empirics-Only Counterfactual Analysis
  ]

  #v(1em)
  #text(size: 12pt)[
    Liming Lin#footnote[Sciences Po Paris. Supervisor: Paul Bouscasse. Jury: Paul Bouscasse, Axelle Ferrière. I am also grateful to Jeanne Commault and to faculty and students at the Department of Economics for their helpful comments. I thank an anonymous friend for facilitating access to a commercial data terminal. All errors are my own.]
  ]

  #v(0.5em)
  May 2026

  #v(0.5em)
  #link("https://github.com/liminglin98/Thesis/blob/reorganized/Rule-Based%20Monetary%20Policy%20in%20China.pdf")[\[Click here for latest version\]]
]

#v(2em)

// Abstract
#align(center)[
  #text(weight: "bold")[Abstract]
]

#block(inset: (left: 0.5in, right: 0.5in))[
  #set par(first-line-indent: 0em)
  Since 2023, Chinese CPI inflation has hovered near zero, persistently undershooting the targets set in the annual Government Work Report. This paper applies the empirics-only counterfactual framework of Caravello, McKay and Wolf (2026) to ask what would have happened to Chinese inflation, output, and the exchange rate if the People's Bank of China had instead followed an explicit targeting rule over this period. Two sets of impulse response functions to monetary policy shocks, one identified from a narrative policy-rule residual on the 7-day repo rate (FR007) and the other from high-frequency FR007 surprises on policy event dates, are combined with a 7-variable Bayesian VAR baseline projection in a stacked least-squares solver to recover the joint shock sequence that best implements alternative targeting rules. The headline finding is a persistent gap between counterfactual CPI paths and the Government Work Report targets, even under strict targeting. The result points to three forces operating together: a genuine multi-objective tradeoff, especially around exchange-rate stabilization; obstacles to conventional rate-based transmission; and a likely post-COVID shift in inflation dynamics that makes the 2023--2025 episode difficult to rationalize using pre-existing macroeconomic relationships alone. 

  #v(0.5em)
  *Keywords:* monetary policy, counterfactuals, China, inflation targeting, sufficient statistics. \
  *JEL codes:* E52, E58, E31.
]

#v(2em)

= Introduction

Since 2023, Chinese CPI inflation has hovered near zero, persistently undershooting the official targets set in the annual Government Work Report (GWR). The GWR CPI targets for 2023, 2024, and 2025 were 3%, 3%, and roughly 2%, respectively; realized year-on-year CPI growth over the same period averaged close to zero. Producer prices have remained in negative territory since late 2022, and the deflationary stance has coincided with severe liquidity stress in the real estate sector, exemplified by the bankruptcies and near-defaults of major developers such as Evergrande and Country Garden, and with elevated local government financing-vehicle (LGFV) debt risks. The episode also differs from earlier easing cycles: whereas the PBoC paired deep benchmark-rate and reserve-requirement cuts with large-scale stabilization efforts in 2008--09 and 2015--16, the post-2022 response was slower and more incremental, despite a comparably visible deterioration in price dynamics.

Despite these pressures, the People's Bank of China (PBoC) has maintained a cautious, gradualist stance. The 1- Year Loan Prime Rate (LPR), now the principal anchor for new loan pricing, has been lowered only moderately, from 3.65% in 2022 to roughly 3.0% in early 2026, a pace that has been widely viewed as insufficient to arrest the deflationary spiral.#footnote[See, for example, the discussion of monetary policy options in the IMF's Article IV consultations with China for 2024 and 2025 @IMFChina2024 @IMFChina2025.] In recent years, the Central Economic Work Conference (CEWC) has introduced the formulation of "a reasonable rebound in the price level" (_wujia heli huisheng_, 物价合理回升); the 2025 CEWC explicitly tied this objective to the conduct of monetary policy. These developments raise a fundamental question about the recent past: was the inflation-target miss mainly a policy choice, or did it reflect the interaction of institutional constraints, weak transmission, and a changed post-COVID macroeconomic environment? Equivalently, what would the trajectory of Chinese CPI, output, and the exchange rate have looked like over 2023--2025 if the PBoC had instead followed an explicit, rule-based targeting policy?

*Approach.* This paper answers that question by applying the empirics-only counterfactual framework of #cite(<CaravelloEtAl2026>, form: "prose") to China.#footnote[Throughout the paper I refer to @CaravelloEtAl2026 as "CMW2026" and to the foundational result in @McKayWolf2023 as "MW2023".] The approach builds on the sufficient-statistics insight of #cite(<McKayWolf2023>, form: "prose"): under weak structural assumptions, evaluating macroeconomic outcomes under alternative monetary policy rules requires only two empirically estimable objects, a reduced-form baseline projection of the economy and the causal impulse-response functions (IRFs) of macro variables to monetary policy shocks. CMW2026 show that, when the contemplated counterfactual involves changes to the short end of the yield curve, these objects can be taken directly from the data, sidestepping the need for a fully specified structural model. The framework is therefore well-suited to settings where the underlying structure is poorly known, arguably the case for China, where the political and economic structures complicate any DSGE-style identification of mechanisms.

The empirical implementation proceeds in four blocks, in the spirit of the CMW2026 application to U.S. post-COVID inflation. First, a 7-variable Bayesian VAR with Minnesota prior, estimated on monthly data from 2002 onward, produces a baseline projection of the Chinese economy from a chosen forecast date. Second, narrative monetary policy shocks are identified as the residual of a forward-looking policy rule for FR007 that combines elements from #cite(<ChenXiaoZha2025>, form: "prose") (asymmetric output gap, lagged interest rate) and #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose") (post-2006 exchange-rate gap term). Third, high-frequency shocks are constructed as daily close-to-close FR007 changes on dates of PBoC announcements that involve a change in one of the core policy instruments (7-day reverse repo, 1-year MLF, 1-year LPR, RRR for large financial institutions, and pre-2015 benchmark deposit/lending rates) and aggregated to monthly frequency. Fourth, the shock sequence that minimizes a weighted quadratic loss over CPI deviations from the GWR target, output deviations from the GWR growth target, FR007 first differences (a smoothness penalty), and policy-induced deviations of the real effective exchange rate from baseline is recovered from a stacked-OLS problem in which both narrative and HFI transmission maps enter as columns of the policy-shock causal-effect matrix.

*Findings.* Two main results emerge.

First, even under strict CPI targeting, in which the loss function places weight only on the inflation gap and an interest-rate smoothness penalty, the counterfactual CPI path remains below the GWR targets throughout 2023--2025. Adding flexible targeting motives (output stabilization, exchange-rate stability) widens the shortfall further. The relevant comparison is therefore between counterfactual CPI and the GWR target, not merely between counterfactual and baseline. The BVAR baseline already falls short of the announced target while still overpredicting realized inflation after 2022, which suggests that the post-COVID episode is difficult to summarize with historical transmission patterns alone. The persistent target gap is the paper's central empirical fact.

Second, the counterfactual reveals a clear policy trade-off, but not a trade-off that can be reduced to discretionary under-reaction alone. The exchange-rate objective is the tightest constraint in the full specification, while the weak and at times perverse CPI response to rate movements implies that conventional short-rate policy is a blunt instrument for closing the inflation gap. The optimizer can move CPI somewhat closer to target only by accepting materially different exchange-rate and output paths.

A related feature of the estimated transmission mechanism is that both FR007-based identification schemes deliver an initial _positive_ CPI response to a contractionary shock, namely a price puzzle. I do not treat this as a separate headline finding; rather, it helps explain why conventional short-rate policy is a blunt tool in the counterfactual exercise. Following #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose"), I interpret the pattern as plausibly reflecting the user-cost methodology of the Chinese CPI's housing component (居住类), through which higher mortgage rates can mechanically raise measured housing costs in the short run. The CMPI-based IRFs attenuate this CPI response and, more importantly, remove the initial positive responses of GDP and IP, suggesting that broader policy disturbances generate a more conventional transmission pattern than FR007 movements alone.

*Contributions.* This paper makes three contributions.

_First_, it is, to my knowledge, the first application of the empirics-only counterfactual framework of #cite(<CaravelloEtAl2026>, form: "prose") to a non-U.S., emerging-market context. The application is non-trivial: the framework was originally designed around the federal funds rate as a single, well-identified policy instrument, while the PBoC operates a multi-instrument framework with dual-track rates. The paper takes a clear stand on this issue by selecting FR007, the market rate that anchors HFI surprises and that #cite(<ChenXiaoZha2025>, form: "prose") use as the dependent variable in their policy rule, as the unifying instrument across all four building blocks.

_Second_, the paper combines two distinct shock-identification schemes within a single counterfactual exercise. Following the CMW2026 logic, both narrative and HFI shocks enter as columns of the transmission map, allowing the solver to recover a joint sequence of historically observed policy surprises rather than privileging one empirical representation ex ante. This is more demanding than the standard practice in the China monetary-policy literature, where a single shock series is typically estimated and compared across alternatives.

The paper also uses the contrast between FR007-based and CMPI-based IRFs to clarify the transmission mechanism. The broader CMPI shocks attenuate the short-run CPI puzzle and remove the initial positive responses of GDP and IP, even though many component instruments are only weakly connected to mortgage pricing. This ancillary result supports the interpretation that the unusual FR007 responses are partly rate-specific rather than a general feature of all Chinese monetary-policy shocks.

*Roadmap.* @sec:literature reviews the related literature on monetary policy identification and counterfactual policy evaluation. @sec:context provides institutional background on the PBoC's policy framework and on the Government Work Report target system. @sec:strategy sets out the empirical framework, including the BVAR baseline projection, the two shock-identification schemes, and the joint counterfactual solver. @sec:data describes the data. @sec:irf presents the estimated impulse responses and discusses the price puzzle. @sec:counterfactual reports the main counterfactual results. @sec:discussion discusses limitations and avenues for future research. @sec:conclusion concludes.

= Chinese Context <sec:context>
== PBoC: From Subordinate Agency to Independent Institution

The PBoC acquired formal central-bank status in 1983 and legal separation from the Ministry of Finance in 1995, after an earlier period in which monetary policy was subordinated to fiscal priorities @ZhengWang2021 @BellFeng2014. Subsequent reforms strengthened its operational capacity: the 1998 branch reform reduced local-government influence, and the separation of supervisory functions in the 2000s allowed the central bank to focus more narrowly on monetary policy and financial stability.

Formal autonomy, however, has never implied full policy independence. Monetary policy authority ultimately remains with the State Council, and the Monetary Policy Committee serves only an advisory role. As #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose") note, substantive policy decisions require State Council approval, limiting the timeliness and forward-looking character of PBoC communication. This institutional setting is the relevant backdrop for the counterfactual exercise: the paper quantifies the implications of a shift toward rule-based targeting, not what the PBoC could have achieved unilaterally within the existing political structure.

== Evolution of Monetary Policy Tools

The transformation of PBoC's policy toolkit over the past three decades reflects both its institutional evolution and a set of structural features specific to China's financial system. In the earlier part of the sample, PBoC relied primarily on quantity-based instruments, especially the Reserve Requirement Ratio (RRR), administratively set benchmark lending and deposit rates, and direct guidance over M2 and total credit growth. The preference for these tools was not simply inertia or imitation of an older Fed playbook; it reflected practical constraints on price-based transmission. China's interest rate markets were heavily repressed for much of this period, with binding ceilings on deposit rates and floors on lending rates that limited the scope for market interest rates to serve as effective policy signals. In an environment where bank lending was allocated partly by administrative guidance and partly through state-owned banks with soft budget constraints, controlling the quantity of credit was a more reliable lever than adjusting its price @DasSong2023. PBoC's mandate compounded this: unlike central banks with a single inflation target, PBoC operated under multiple official objectives, including price stability, economic growth, employment, exchange rate stability, and financial market development, which favored a toolkit broad enough to address each dimension separately @MirandaAgrippinoNenovaRey2026. M2 growth served as the primary intermediate target through the mid-2010s, with the RRR functioning as a blunt but effective instrument for managing aggregate bank liquidity. From 2012, as the rise of shadow banking eroded the correspondence between M2 and broader credit conditions, PBoC increasingly emphasized Total Social Financing (TSF), which better captured off-balance-sheet and non-bank lending @ChenRenZha2018. During the WTO-era surplus years, Central Bank Bills were added as a sterilization instrument to absorb the excess liquidity generated by foreign exchange inflows, and were phased out after 2013 as the exchange rate regime was liberalized and the sterilization motive diminished.

The transition toward price-based policy accelerated from 2013 onward, alongside a recognition by PBoC itself that the correlation between M2 and real activity was weakening as the economy became more market-oriented @YiGang2018PressConference; this transition was later reinforced by the broader financial liberalization agenda of the 13th Five-Year Plan. PBoC introduced the Standing Lending Facility (SLF) and the Medium-Term Lending Facility (MLF) to manage short- and medium-term liquidity respectively, with the SLF rate forming the corridor ceiling. The completion of deposit rate liberalization in October 2015 and the progressive elevation of the 7-day reverse repo rate (FR007) as the operational short-term anchor marked the emergence of a two-tier interest rate corridor. The reform of the Loan Prime Rate (LPR) in August 2019, reconstituting it as a spread over the 1-year MLF rate, extended price-based transmission into the retail credit market. By the latter part of the sample, the composite monetary policy indicator (CMPI) of #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose") co-moves primarily with FR007, the 3-month SHIBOR, and the LPR, while RRR adjustments continue as a supplementary structural liquidity tool. #cite(<ChenXiaoZha2025>, form: "prose") and #cite(<DasSong2023>, form: "prose") both build their shock identification around FR007 (and its Interest Rate Swap) movements, reflecting its status as the primary observable policy rate in the post-2015 framework.

Despite the move toward price-based tools, China remains a multi-instrument system. Structural facilities such as the Pledged Supplementary Lending program, re-lending, and re-discounting increasingly direct subsidized liquidity toward preferred sectors, blurring the boundary between monetary and fiscal policy and complicating the interpretation of any single observable policy indicator. Those identification issues return below; for the present paper, the key implication is that FR007 must be treated as a summary of the short-rate stance rather than as the whole policy regime.

== The Government Work Report Target System

The Government Work Report (GWR), delivered by the Premier to the National People's Congress each March, sets annual numerical objectives for GDP growth, CPI inflation, employment, and the fiscal deficit. These are not legally binding commitments: the PBoC's statutory mandate contains no quantitative CPI target. The GWR targets instead function as an implicit accountability device, providing a public benchmark for evaluating policy priorities. The counterfactual therefore asks not what would have happened under a legally embedded inflation-targeting regime, but what would have happened had the PBoC treated the government's own announced objectives as binding operational constraints.

The Central Economic Work Conference (CEWC), convened each December, sets the agenda that the subsequent GWR operationalizes. Its recent language on a "reasonable rebound in the price level" and on monetary support for that objective helps justify treating the announced CPI benchmark as economically meaningful rather than purely rhetorical. The 2025 GWR lowered the CPI target from "around 3%" to "around 2%," retained in 2026; the loss function takes the stated target as the operative benchmark.

#figure(
  image("outputs/figures/quarterly_real_gdp_growth_vs_target.png", width: 92%),
  caption: [Official quarterly real GDP growth and the Government Work Report GDP target, 2002–2025.],
) <fig:gdp-target-comparison>

One further reason to use the GWR benchmarks rather than a conventional Taylor rule is that both the inflation and growth targets are observable. The loss function can therefore be built from public targets rather than from estimates of unobservable potential output or the natural rate of interest. In that sense, the exercise is closer to evaluating an announced nominal-income-style benchmark than to estimating a conventional Taylor rule.

#figure(
  image("outputs/figures/monthly_cpi_vs_target.png", width: 92%),
  caption: [Monthly CPI inflation and the Government Work Report CPI target, 2002-2025.],
) <fig:cpi-target-comparison>

= Literature Review on Macroeconomic Policy Counterfactuals <sec:literature>

Evaluating the macroeconomic effects of monetary policy faces two interlocking problems. First, policy responds endogenously to the state of the economy, so the observed correlation between policy variables and macroeconomic outcomes cannot be interpreted causally without identifying restrictions. Second, even if the causal effects of unanticipated policy actions can be identified, the relationships between policy and outcomes are not generally invariant to changes in the systematic component of policy, which is the #cite(<Lucas1976>, form: "prose") critique. A counterfactual exercise is meaningful only insofar as the relationships used to construct it remain valid under the contemplated change in the policy rule.

Two parallel literatures have responded to these problems. The first, organized around vector autoregressions (VARs), seeks to identify policy shocks while remaining as agnostic as possible about deep structural parameters. The second, organized around the New Keynesian (NK) research program, builds fully specified microfounded models in which the structural parameters are by construction invariant to the policy rule. A recent third line of work bridges these approaches by establishing sufficient-statistics conditions under which counterfactuals can be obtained from reduced-form objects alone. This paper sits within the third tradition and applies its empirics-only variant to China.

== The reduced-form VAR tradition

The modern empirical literature on monetary policy traces to #cite(<Sims1980>, form: "prose"), who proposed VARs as an alternative to the large-scale structural macroeconometric models that had dominated the field, on the grounds that their identifying restrictions were "incredible." Within this tradition, a substantial sub-literature has developed methods to identify exogenous variation in policy. Two principal approaches have emerged. Narrative methods, originating in #cite(<RomerRomer2004>, form: "prose"), residualize policy decisions against the systematic policy rule by exploiting documentary evidence on policy intentions; high-frequency identification (HFI) methods, originating in #cite(<Kuttner2001>, form: "prose") and extended in #cite(<GSS2005>, form: "prose") and #cite(<NakamuraSteinsson2018>, form: "prose"), exploit movements in market rates within tight windows around policy announcements. The identified shock series are typically used as external instruments in SVAR settings @MertensRavn2013 @StockWatson2018 to recover impulse response functions.

Both shock-identification approaches face ongoing challenges. The most prominent is the information-effect critique: that even properly identified policy shocks may convey private central-bank information about future economic conditions, so the responses of macro variables to "shocks" reflect both the direct policy impulse and an information channel @NakamuraSteinsson2018 @JarocinskiKaradi2020. Recent work pushes back on this interpretation, arguing that what appears as an information effect is largely a response to predictable economic news @MirandaAgrippino2021 @BauerSwanson2023. Because both methods are central to the methodology of this paper, Section 4 returns to them in greater detail and surveys the literature on their application to China.

Once policy shocks are identified, the natural next step in the VAR tradition is to construct counterfactual paths by manipulating the shock sequence in an estimated SVAR. #cite(<BGW1997>, form: "prose") implement this approach for oil-price shocks, asking how much of the post-shock decline in U.S. output is attributable to the systematic monetary response rather than to oil prices directly. #cite(<SimsZha1995>, form: "prose") develop the canonical multi-variable identified VAR and use it to evaluate alternative U.S. policy regimes. The principal difficulty with this strategy is that VAR-based counterfactuals constructed by repeatedly drawing policy shocks to implement an alternative rule remain vulnerable to the Lucas critique. If private agents had expected the alternative rule to prevail, the reduced-form coefficients of the VAR, including the relationship between shocks and outcomes, would generally differ. Reduced-form counterfactuals are thus best understood as informative about modest deviations from the prevailing rule, rather than as evaluations of large or persistent regime changes.

== The structural NK tradition

The parallel response to the Lucas critique builds models in which agents optimize given the policy rule, so that the parameters describing preferences and frictions are invariant by construction. Medium-scale New Keynesian DSGE models became the workhorse for monetary-policy counterfactuals, while later HANK work showed that transmission can change materially once household heterogeneity and incomplete markets are modeled explicitly @CEE2005 @SmetsWouters2007 @KMV2018 @AuclertEtAl2021. The strength of the structural approach is regime invariance; its cost is model dependence, since different assumptions about frictions and heterogeneity can imply different counterfactual answers.

== Sufficient statistics and the empirics-only revival

A recent line of work bridges these two traditions. #cite(<McKayWolf2023>, form: "prose") establish that, under invertibility, the impulse responses of macroeconomic variables to a set of contemporaneous and anticipated ("news") shocks to the prevailing policy rule, together with a baseline projection of the economy, constitute a sufficient statistic for counterfactual policy evaluation under a wide class of alternative rules. The key idea is that the counterfactual is constructed not by changing the policy rule and re-propagating through the estimated reduced-form dynamics, the strategy of #cite(<SimsZha1995>, form: "prose"), which encounters the Lucas critique because the reduced-form dynamics themselves depend on the prevailing rule, but by implementing the alternative policy path through a sequence of policy and news shocks, all occurring within the prevailing rule. Under linearity, this within-regime construction is mathematically equivalent to the across-regime counterfactual to first order, and the empirically estimated IRFs already encode the correct private-sector response. The validity of this equivalence depends on the contemplated counterfactual remaining within the span of historically observed policy movements; the scope of this assumption is discussed in Section 4. #cite(<CaravelloEtAl2026>, form: "prose") extend this insight by showing that, when the contemplated counterfactual involves changes only at the short end of the yield curve, the required IRFs can be estimated purely empirically, sidestepping the need to commit to a particular structural model. #cite(<BouscasseHong2024>, form: "prose") apply a closely related sufficient-statistics framework to evaluate counterfactual fiscal rules in the United States.

The sufficient-statistics literature thus offers an alternative to both horns of the structural-vs-reduced-form trade-off: it retains the empirical discipline of the VAR tradition while reducing the need to commit to a full structural model. Its empirical demands remain substantial, however, because reliable policy IRFs are still required in settings where instruments are multiple and expectations are difficult to observe.

== Position of this paper

This paper applies the empirics-only framework of #cite(<CaravelloEtAl2026>, form: "prose") to evaluate counterfactual Chinese monetary policy over the 2023-2025 deflationary episode. The exercise serves two purposes. First, it stress-tests the framework outside its original U.S. context, in a setting where policy is conducted through a multi-instrument operating framework, where fiscal-monetary coordination is institutionalized, and where the structural transmission of policy is incompletely understood, which are conditions that arguably make the empirics-only approach more attractive than the structural alternative, but that also raise concerns about whether its identification requirements can be met. Second, it produces a substantive answer to the question of whether a rule-based PBoC could have closed the realized inflation gap relative to the announced Government Work Report targets, or whether the realized inflation undershoot reflects deeper structural constraints on monetary policy in China.

The identification of Chinese monetary policy shocks, which feed the framework, draws on a parallel literature that has adapted both narrative and high-frequency methods to the Chinese institutional setting. That literature covering work by #cite(<ChenRenZha2018>, form: "prose"); #cite(<ChenXiaoZha2025>, form: "prose"); #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose"); #cite(<DasSong2023>, form: "prose"); #cite(<HeEtAl2023>, form: "prose"); and #cite(<GurkaynaklpekRicco2026>, form: "prose"), among others is reviewed in Section 4 alongside the implementation details of the methodology.

= Empirical Strategy <sec:strategy>

== The CMW Sufficient-Statistics Approach

The sufficient-statistics approach of #cite(<McKayWolf2023>, form: "prose") and #cite(<CaravelloEtAl2026>, form: "prose") (henceforth CMW) requires two empirical objects: a baseline projection and policy IRFs. Operationally, the counterfactual path is the baseline forecast plus a transmission map applied to a sequence of policy shocks chosen to satisfy an alternative rule. The interpretation remains within-regime: the IRFs are held fixed, so the contemplated policy deviation must stay within the span of historically observed policy movements rather than represent a wholesale regime change.

=== Core Equations

CMW formalize the sufficient-statistics result with two objects. First, under the prevailing policy regime, the observed vector of macroeconomic variables admits the Wold representation

$ y_t = sum_(ell = 0)^infinity Psi_ell u_(t - ell), $

where $u_t$ denotes orthogonalized one-step-ahead forecast errors. Second, allowing arbitrary wedges $nu$ in the policy rule defines the policy-transmission map

$ y = Theta_nu nu, $

where $Theta_nu$ collects the paths of macroeconomic outcomes implementable through policy shocks at all horizons. Proposition 1 of CMW (2026) shows that, if the underlying SVMA process is invertible, then knowledge of $Theta_nu$ together with the Wold representation is sufficient to recover the counterfactual process under an alternative monetary-policy rule and its implied second moments; their discussion in §2.2 clarifies that invertibility matters here because it ensures that the Wold representation delivers the relevant forecasts. For the purposes of this paper, the implication is simple: once baseline forecasts and policy causal effects are known, the counterfactual can be constructed without fully specifying the rest of the economy.

=== Assumptions and Their Validity in the Chinese Context

The CMW framework rests on several maintained assumptions, each of which is non-trivial in China.

*Fiscal environment.* The framework assumes a fixed fiscal rule and a unified fiscal authority. China violates both conditions: fiscal institutions changed materially over the sample, and local governments account for a large share of spending while monetary-fiscal coordination is organized through the State Council. Estimated IRFs may therefore absorb correlated fiscal responses rather than monetary transmission alone.

*No information effects.* Monetary shocks should reflect policy transmission rather than private information about fundamentals. Whether PBoC actions contain such information effects remains open; the issue is revisited in §6.

*Symmetric effects.* The framework assumes that expansionary and contractionary shocks have mirror-image responses. The sign-split exercises in the appendix suggest approximate symmetry for the narrative, HFI, and CMPI specifications.

*Single policy instrument.* The main specification treats FR007 as a sufficient summary of the short-rate stance in a multi-instrument framework. The counterfactual should therefore be read as asking what would have happened under a different interbank-rate path, not as identifying the effects of each PBoC instrument separately. CMPI-based results provide a robustness check in §5.4.1.


== Narrative Shock Identification

=== 4.2.1 Genealogy

The #cite(<RomerRomer2004>, form: "prose") approach, regressing the policy rate on the central bank's own forecasts and extracting residuals, does not translate directly to China for two reasons: the PBoC does not publish macroeconomic forecasts comparable to the Fed's Greenbook, and there exist no authoritative estimates of potential output or the natural rate of interest for China. The Government Work Report (GWR) targets serve as a pragmatic substitute for these unobservable benchmarks.

Several papers have adapted the narrative approach to the Chinese context. Table 1 summarizes the key methodological choices across specifications.

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header(
    [],
    [Chen, Xiao & Zha (2025)],
    [Miranda-Agrippino et al. (2026)],
    [This paper],
  ),
  [*LHS variable*], [FR007], [CMPI], [FR007],
  [*RHS timing*], [Contemporaneous], [Lagged ($t - 1$)], [Lagged ($t - 1$)],
  [*Exchange rate gap*], [No], [Yes (CPR × spot-parity gap)], [Yes],
  [*Asymmetric GDP response*], [Yes (threshold from CRZ 2018)], [Yes (threshold ≈ 1pp)], [Yes (threshold ≈ 1pp)],
  [*Sample*], [2002-2019], [2002-2019], [2002-2022],
)

#cite(<ChenRenZha2018>, form: "prose") (henceforth CRZ), updated in #cite(<ChenXiaoZha2025>, form: "prose"), estimated a monetary policy rule for China using the reserve requirement ratio (and later FR007) as the dependent variable, with an important innovation: an asymmetric, regime-switching response to the GDP gap, reflecting the PBoC's greater tolerance for above-target growth than below-target growth.

MANR (2026) introduced the Composite Monetary Policy Indicator (CMPI) on the left-hand side of the regression, encompassing 36 policy instruments with time-varying equal weights (see Table A.1 in MANR). They further added an exchange rate gap term and used lagged ($t - 1$) right-hand-side variables, which mechanically makes the policy rule forward-looking, though with the implicit assumption of perfect one-period foresight. MANR estimated the asymmetric GDP-gap threshold at approximately 1 percentage point, meaning that the PBoC tolerates small overshoots in output growth. Notably, only the coefficient on the negative GDP gap (below-target growth) is statistically significant, a finding we replicate in our FR007 specification (Appendix @tab:fr007-policy-rule). MANR also document a positive CPI response to contractionary monetary policy shocks, which they attribute to the user-cost methodology of Chinese housing CPI, a pattern we obtain under both narrative and HFI identification and discuss as a substantive finding in the results section.

=== Our Specification

We combine elements from CXZ (2025) and MANR (2026). The monthly policy rule for the 7-day interbank repo rate (FR007) is:

$ "FR007"_t = alpha + rho dot "FR007"_(t - 1) + beta_pi dot "cpi_gap"_(t - 1)
  quad + beta_y^(+) dot "gap_pos"_(t - 1) + beta_y^(-) dot "gap_neg"_(t - 1) \
  quad + beta_e dot "fx_gap"_(t - 1) + epsilon_t^"MP" $

where:

- $"cpi_gap"_t = pi_t - pi^*_t$, with $pi^*_t$ the annual CPI target from the Government Work Report (stepping in March at each NPC announcement, not January).
- $"gdp_gap"_t = g_t - g^*_t$, where $g_t$ is a monthly real GDP proxy constructed via the #cite(<StockWatson2010>, form: "prose") distribution method(which will be discussed in more details in the data section and appendix), and $g^*_t$ is the GWR growth target.
- $"gap_pos"_t$, $"gap_neg"_t$: asymmetric split of the GDP gap around a 1pp threshold.
- $"fx_gap"_t$: a post-2006 indicator (after the end of the fixed exchange rate regime) interacting the change in the central parity rate with the spot–parity deviation.

The residual $hat(epsilon)_t^"MP"$ is the narrative monetary policy shock series.


#figure(
  image("outputs/robustness/2025/FR007_residuals_comparison.png", width: 92%),
  caption: [Narrative monetary policy shock series from the FR007 policy-rule residuals, 2002–2025.],
) <fig:narrative-shock-series>


=== Why FR007 as the Policy Indicator

We retain FR007 rather than the CMPI on the left-hand side for two reasons. First, FR007 is the actual market rate that anchors the HFI surprises downstream, maintaining internal consistency between the two shock identification strategies. Second, while the official 7-day reverse repo rate became the PBoC's primary signaling tool only after 2015, the market FR007 has reflected interbank liquidity conditions and responded to all PBoC operations throughout the sample, including RRR adjustments and benchmark rate changes in the pre-2015 era. FR007 is not the policy instrument per se in the early sample; rather, it transmits whatever instrument the PBoC was using at the time, consistent with the use of FR007 in #cite(<ChenXiaoZha2025>, form: "prose"). Appendix @tab:fr007-pre2015-policy-response documents this pre-2015 responsiveness.

=== Critique of the CMPI Approach

While the CMPI solves the problem of an ever-changing policy toolkit by aggregating all instruments into a single index, three features limit its usefulness for our counterfactual exercise.

First, *interpretability*: the counterfactual shock $tilde(nu)$ applied to the CMPI lacks a transparent economic interpretation. A one-unit CMPI shock does not correspond to any specific policy action; it is a weighted average of deviations from trend across heterogeneous instruments. In contrast, a shock to FR007 maps directly onto an interbank rate movement.

Second, *cross-validation with HFI*: most PBoC policy tool changes are announced outside normal trading hours, so the market reaction to changes in individual tools cannot be cleanly isolated in daily data using a close-to-close HFI window. The CMPI, by construction, aggregates these tools and therefore cannot be externally validated by high-frequency market surprises in the way that FR007 can.

Third, *equal weights are atheoretical*: MANR assigns $1 / abs(N_t)$ weight to each instrument active at date $t$, where $abs(N_t)$ is the number of active instruments. This treats a 25bp RRR cut, which directly affects trillions of yuan in required reserves, identically to a 10bp adjustment in an obscure short-term lending facility rate. There is no theoretical or empirical basis for this assumption. A potential improvement would be to weight instruments by their associated liquidity impact or to use a factor model to extract a common policy component, but we leave this for future work.


== High-Frequency Identification

=== Background

High-frequency identification of monetary policy shocks exploits the assumption that, within a narrow window around a policy announcement, movements in market interest rates are dominated by the surprise component of the policy action. This approach, developed for the US context by #cite(<Kuttner2001>, form: "prose"), #cite(<GSS2005>, form: "prose"), and others, has been adapted to China by #cite(<DasSong2023>, form: "prose"), who used daily changes in interest rate swaps (IRS) on FR007 around PBoC announcement dates. #cite(<HeEtAl2023>, form: "prose") propose a two-stage HFI procedure: in the first stage, they purge raw market surprises of any PBoC information effects by projecting them onto a broad set of macroeconomic releases; in the second stage, the cleaned surprises serve as instruments in a structural VAR. This two-stage design addresses the concern that central bank announcements convey information about economic conditions beyond the policy action itself. A distinct approach uses London copper prices as an outside instrument for Chinese monetary policy, exploiting China's dominant role in global commodity demand (#cite(<GurkaynaklpekRicco2026>, form: "prose")).



=== Our HFI Specification

We identify monetary policy surprises as the daily close-to-close change in FR007 on dates when the PBoC announces a change to one of its core policy instruments. In contrast to studies using FR007 IRS quotes, I use daily spot FR007 because a consistent historical IRS series was not available for the full sample:

$ "shock"_t = ("FR007"_t - "FR007"_(t - 1)) dot bold(1)_(t in cal(E)) $

where $cal(E)$ is the set of policy event dates. The core instruments tracked are:

- 7-day reverse repo rate (OMO), the primary short-term policy rate post-2016
- 1-year Medium-term Lending Facility (MLF), introduced in 2014
- 1-year Loan Prime Rate (LPR), which replaced the benchmark lending rate in August 2019
- Reserve Requirement Ratio for large financial institutions (RRR)
- Benchmark deposit and lending rates (pre-2015, before rate liberalization)

Event dates are sourced from PBoC announcements collected via Wind. We restrict to dates on which a rate _change_ occurs; dates with announcements confirming no change are excluded because they generate minimal market surprise in the close-to-close window.

Daily shocks are summed to monthly frequency for use in the BVAR. This summation maintains a linearity assumption: transmission is additive across within-month shocks. If transmission is state-dependent, sign-asymmetric, or size-nonlinear, the monthly aggregation may introduce measurement error. We flag this as a maintained assumption rather than attempt to address it within the current framework.


#figure(
  image("outputs/figures/monthly_hfi_policy_change_shock_series.png", width: 92%),
  caption: [Monthly rate-change-only HFI shock series, 2002–2025.],
) <fig:hfi-policy-change-shocks>


=== Complementarity of Narrative and HFI Shocks

The two shock series serve distinct identification roles. The narrative shock is identification-by-construction: residuals from an estimated policy rule, capturing the unsystematic component of interest rate movements after controlling for the PBoC's reaction to inflation, growth, and exchange rate gaps. The HFI shock is identification-by-window: the market surprise within a narrow event window, with no assumption about the policy reaction function. The two approaches are complementary, not substitutes, and it is this complementarity that motivates using both as inputs to the counterfactual. 

=== Limitations and Potential Improvements

Several limitations of our HFI implementation merit acknowledgment.

_IRS vs. raw FR007._ Ideally, the HFI surprise would be measured using changes in the 1-year interest rate swap on FR007, which is more forward-looking and less affected by daily PBC liquidity operations than raw FR007. #cite(<DasSong2023>, form: "prose") use IRS for this reason. We use raw FR007 due to data access constraints (full hourly IRS history requires access to the China Foreign Exchange Trade System).


_Mid-month contamination._ PBoC policy announcements, particularly MLF operations, tend to cluster around the 15th of each month, coinciding with NBS releases of major macroeconomic indicators (CPI, industrial production, etc.). This co-timing raises the concern that the HFI surprise reflects the market's joint reaction to the policy action and the macro data release, rather than the policy surprise alone. The Wind economic calendar available in our files begins only on January 25, 2007, so it does not cover the full HFI sample; within the covered period, @tab:hfi-policy-macro-overlap compares the broad all-announcement series with the preferred rate-change-only series. Addressing this contamination rigorously would require intraday IRS data and a complete macroeconomic calendar for the full sample period.

== Impulse Response Estimation

The impulse responses used to populate the transmission maps ($Pi$ matrices) are estimated via two Bayesian SVARs with instrumental-variable identification. The narrative shock residuals $hat(epsilon)_t^"MP"$ serve as the external instrument for the FR007 equation in one estimation, while the monthly-aggregated HFI series serves as the external instrument in the other. The resulting two IRF sets jointly define the baseline policy-transmission map used in the counterfactual exercise.

Both identification schemes yield a positive short-run CPI response to contractionary monetary policy shocks. We return to its interpretation below.


== Baseline Economic Forecast

The baseline forecast is constructed from a 7-variable Bayesian VAR with 6 lags estimated at monthly frequency. The variables are: monthly real GDP (constructed via Stock and Watson 2010 interpolation), CPI year-on-year, FR007, M2 year-on-year growth, RMB REER year-on-year, US industrial production year-on-year, and industrial value added year-on-year. The estimation sample runs from 2002 to 2022 for the primary specification, with a 2002-2025 sample used as a robustness check. COVID dummies are included in all specifications.


The estimated monthly BVAR is:

$ y_t = c + sum_(ell = 1)^6 A_ell y_(t - ell) + sum_(k = 1)^4 d_k D_t^("covid", k) + u_t, $

where $D_t^("covid", k)$ are monthly dummies for January--April 2020. We use a Minnesota-type prior following #cite(<GiannoneLenzaPrimiceri2015>, form: "prose"), as adopted by MANR (2026). The Minnesota prior addresses parameter proliferation, with 7 variables and 6 lags, the system has 43 regressors per equation, by shrinking coefficients toward a random-walk prior (own first lag = 1, all other coefficients = 0), with tightness declining harmonically in the lag order. This is the standard motivation for Bayesian shrinkage in richly parameterized macro VARs.

CMW (2026) use the language of the Wold representation to establish the theoretical validity of the counterfactual (Proposition 1). In practice, the computational object is simply the iterated BVAR forecast, which is equivalent to the Wold representation under invertibility. Since the counterfactual projection begins after 2020, the COVID dummies are zero over the forecast horizon. The baseline projection at horizon $h$ from date $t^*$ is therefore:

$ hat(y)_(t^* + h) = hat(c) + sum_(ell = 1)^6 hat(A)_ell hat(y)_(t^* + h - ell), quad h = 1, 2, dots, H $

with $H = 120$ months (10 years).

== Counterfactual Construction

=== Setup

The counterfactual combines three objects from the preceding sections: baseline forecast paths $pi^"base", y^"base", i^"base"$ over a $T$-month horizon; transmission maps $Pi_pi, Pi_y, Pi_i$ linking both narrative and HFI shocks to each variable (constructed as Toeplitz matrices from the estimated IRFs); and policy targets $pi^*, y^*$ from the Government Work Report.

We solve for one stacked shock vector,

$ tilde(nu) =
  mat(
    tilde(nu)^N;
    tilde(nu)^H
  )
  in RR^(2T), $

where the top $T$ entries are narrative shocks and the bottom $T$ entries are HFI shocks. Counterfactual paths are the baseline plus the transmission:

$ tilde(pi) = pi^"base" + Pi_pi tilde(nu), quad tilde(y) = y^"base" + Pi_y tilde(nu), quad tilde(i) = i^"base" + Pi_i tilde(nu) $

Each transmission map horizontally concatenates two lower-triangular Toeplitz blocks, one for each identification scheme:

$ Pi_pi = [Pi_pi^N | Pi_pi^H], quad Pi_pi tilde(nu) = Pi_pi^N tilde(nu)^N + Pi_pi^H tilde(nu)^H, $

with the same construction for output, FR007, and the exchange rate. Thus each $Pi$ matrix is $T times 2T$: within each block, the $(t, s)$ entry gives the response at horizon $t - s$ to a unit shock at date $s$.

=== Loss Function

The shock sequence is chosen to minimize:

$ cal(L)(tilde(nu)) = lambda_pi abs(tilde(pi) - pi^*)^2 + lambda_y abs(tilde(y) - y^*)^2 + lambda_i abs(Delta tilde(i))^2 + lambda_e abs(tilde(e))^2 $

The four terms penalize: (i) CPI deviation from the GWR inflation target; (ii) GDP deviation from the GWR growth target; (iii) abrupt changes in FR007 (interest-rate smoothing); and (iv) policy-induced exchange rate instability. The weights $lambda_pi, lambda_y, lambda_i, lambda_e$ define the targeting rule. Following CMW (2026), the discount factor is set to 1. This formulation is algebraically equivalent to the rule-satisfaction minimization in Caravello et al. (2026), up to a sign convention on the shock vector.

We consider three targeting rules:

1. *CPI only*: $lambda_pi = 1, lambda_y = 0, lambda_i = 1, lambda_e = 0$
2. *CPI + GDP*: $lambda_pi = 1, lambda_y = 1, lambda_i = 1, lambda_e = 0$
3. *CPI + GDP + REER*: $lambda_pi = 1, lambda_y = 1, lambda_i = 1, lambda_e = 1$

=== Solution

Because the counterfactual paths are linear in $tilde(nu)$ and the loss is quadratic, the problem reduces to a weighted least-squares system:

$ tilde(nu)^* = arg min_(tilde(nu)) cal(L)(tilde(nu)) = A_"stack" backslash b_"stack" $

where

$ A_"stack" =
  mat(
    sqrt(lambda_pi) Pi_pi;
    sqrt(lambda_y) Pi_y;
    sqrt(lambda_i) D Pi_i;
    sqrt(lambda_e) Pi_e
  ),
  quad
  b_"stack" =
  mat(
    sqrt(lambda_pi) (pi^* - pi^"base");
    sqrt(lambda_y) (y^* - y^"base");
    sqrt(lambda_i) (d_"anchor" - D i^"base");
    sqrt(lambda_e) (0 - e^"base")
  ). $

Here $D$ is the anchored first-difference operator for FR007, with $d_"anchor" = (i_"last", 0, dots, 0)'$ ensuring continuity from the last observed interest rate into the first counterfactual month. Stacking works because each term in the original loss can be written as the squared norm of one weighted linear residual in $tilde(nu)$; concatenating those residuals gives

$ cal(L)(tilde(nu)) = abs(A_"stack" tilde(nu) - b_"stack")^2, $

so minimizing the original policy objective is exactly equivalent to solving one ordinary least-squares problem. Because each transmission map has $2T$ columns, $A_"stack"$ operates on the same joint $2T$-vector of narrative and HFI shock choices. The Julia implementation follows this stacked-system construction directly.

For a three-month horizon, the inflation transmission map has the schematic form

$ Pi_pi =
  [
    underbrace(
      mat(
        theta_0^N, 0, 0;
        theta_1^N, theta_0^N, 0;
        theta_2^N, theta_1^N, theta_0^N
      ),
      "narrative shocks"
    )
    |
    underbrace(
      mat(
        theta_0^H, 0, 0;
        theta_1^H, theta_0^H, 0;
        theta_2^H, theta_1^H, theta_0^H
      ),
      "HFI shocks"
    )
  ],
  quad
  tilde(nu) =
  mat(
    tilde(nu)_1^N;
    tilde(nu)_2^N;
    tilde(nu)_3^N;
    tilde(nu)_1^H;
    tilde(nu)_2^H;
    tilde(nu)_3^H
  ). $

Multiplication by the joint shock vector therefore adds the narrative-shock contribution and the HFI-shock contribution month by month, while preserving the lag structure implied by each IRF sequence.

= Data Construction <sec:data>

== Monthly GDP Construction

Because official real GDP is available only quarterly, we construct a monthly proxy using the #cite(<StockWatson2010>, form: "prose") distribution method. Quarterly GDP is allocated across months using expenditure-side monthly indicators and a state-space model that preserves exact quarterly aggregation. The full indicator set, implementation, and validation checks are reported in @app:monthly-gdp.

#figure(
  image("outputs/diagnostics/gdp_quarterly_vs_proxy_monthly.png", width: 92%),
  caption: [Reported quarterly real GDP growth and the Stock-Watson distributed monthly GDP proxy.],
) <fig:gdp-proxy-validation>

_Robustness to GDP construction._ Because the inferred monthly path depends on the indicator set used for distribution, @app:monthly-gdp reports the decomposition and validation checks.

#line(length: 100%)

== Baseline Forecast BVAR Variables

The 7-variable BVAR is estimated at monthly frequency with the following variable ordering:

$ y_t = (g_t, "IP"_t, pi_t, i_t, m_t, e_t, y_t^("US"))' $

The variable set balances the need to represent output, prices, monetary conditions, and the external environment against Chinese data constraints. GDP is the loss-function output measure, IVA provides an independent monthly activity signal, M2 captures the quantity dimension of transmission, the REER captures the external channel, and U.S. industrial production controls for foreign demand. Full variable definitions, transformations, and sources are reported in @app:bvar-vars.


== Policy Targets from the Government Work Report

The GWR CPI and GDP growth targets define the counterfactual benchmarks ($pi^*_t$, $g^*_t$). Targets step in March, ranges are coded at their midpoint, and years without a newly announced target carry forward the previously operative benchmark so that the target series remains continuous. The complete target construction is documented in @app:gwr-targets. For the narrative rule, the exchange-rate gap uses the bilateral CNY/USD rate because a full-sample CFETS-basket measure is not available.


== CMPI Reconstruction

The Composite Monetary Policy Indicator (CMPI) of #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose") aggregates deviations from trend across the PBoC's full instrument toolkit into a single index. We reconstruct it through 2025 using the MANR instrument categories; the full extracted panel is reported in the appendix.

The CMPI is used only for robustness; the main counterfactual uses the joint FR007-based transmission map built from narrative and HFI shocks. The reason is that the counterfactual requires shocks on a variable with transparent economic interpretation: a one-unit FR007 shock maps to a basis-point interbank rate movement, while a one-unit CMPI shock is a weighted average across heterogeneous instruments with no direct policy interpretation. The CMPI enters the BVAR in the robustness specification as an additional variable, with a separate policy rule estimated on CMPI instead of FR007; the resulting shock series is compared with the FR007 narrative shock in @fig:cmpi-vs-fr007-narrative-shocks.

= Results

== Impulse Responses <sec:irf>

=== Narrative Identification

Figure @fig:narrative-irfs reports impulse responses of monthly GDP, CPI, the real effective exchange rate (REER), and industrial production (IP) to a 1 percentage point contractionary shock to FR007, identified using the narrative residuals from the policy rule. The shock is normalized so that FR007 rises by 1pp on impact.

Real GDP exhibits a brief positive spike of approximately 0.25pp in the first month before declining to zero around the fourth month, reaching a trough of roughly -0.5pp near the tenth month, and recovering gradually thereafter. The initial positive response is consistent with two non-exclusive interpretations: a residual information effect, in which a rate hike signals confidence in underlying growth, and a monthly-frequency timing effect, in which output is still expanding within the month when the PBoC tightens.

CPI rises to a peak of approximately 0.25pp around the fifth month before declining through zero near the fourteenth month and reaching a trough at roughly −0.125pp. This positive CPI response to a contractionary shock is a price puzzle that mirrors the finding in #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose"), who report a peak of approximately 0.5pp with a zero crossing around the twelfth month using CMPI-based identification. We interpret the puzzle not as a misspecification but as reflecting the user-cost methodology of the Chinese CPI's housing component (居住类): compared with the U.S. fixed-rate mortgage system, Chinese housing costs are more exposed to policy-rate movements, which can raise measured housing costs in the short run through mortgage and rent channels. This interpretation is corroborated by the CMPI-based IRFs (Appendix @app:cmpi-irfs), where the use of a broader policy indicator that captures non-rate instruments, which do not directly affect mortgage costs, yields a substantially smaller initial CPI increase and a more conventional negative response within five months.

The real effective exchange rate appreciates, consistent with standard theory: the REER rises to a peak of approximately 1.25pp around the tenth month before reverting toward zero. Industrial production, included both as a robustness check on the GDP response and as an input to the counterfactual, displays a qualitatively similar pattern, with an initial spike to 0.5pp around the third month, a zero crossing at the fourth month, and a trough near $-$1pp around the thirteenth month. The IP response is roughly double the GDP response in magnitude, which likely reflects the higher cyclical sensitivity of industrial output relative to services-inclusive GDP in the Chinese economy.

#figure(
  image("outputs/main_results/2022/irf_bvar_iv_svar.png", width: 92%),
  caption: [Impulse responses to a 1 percentage point contractionary narrative FR007 shock, BVAR estimated through 2022.],
) <fig:narrative-irfs>


=== High-Frequency Identification

Figure @fig:hfi-irfs reports the corresponding IRFs using the rate-change-only HFI shock series. The medium-run dynamics are broadly consistent with the narrative identification: GDP reaches a trough of approximately −0.5pp near the tenth month, CPI peaks at 0.6pp around the fifth month and crosses zero near the sixteenth month, and the REER peaks at 1.75pp around the tenth month. The principal discrepancy is in the short run. GDP and IP exhibit substantially larger initial spikes (approximately 2pp and 1.5pp, respectively) under HFI identification, which we attribute to known measurement issues in the monthly HFI series: contamination from the overlap between PBoC announcement dates and NBS macro data releases (see §4.3 and Table @tab:hfi-policy-macro-overlap), and the noise introduced by aggregating daily close-to-close changes to monthly frequency.

The convergence of medium-run responses across two independent identification strategies, one exploiting the systematic policy rule, the other exploiting announcement-day market movements, supports combining them within the counterfactual transmission map. The narrative series is more stable in the short run, while the HFI series contributes a complementary announcement-window measure of policy surprises. The resulting joint map should still be interpreted cautiously because the HFI block is noisier at short horizons, but the common medium-run pattern is precisely what makes the two approaches useful together rather than merely as substitutes.

#figure(
  image("outputs/main_results/2022/irf_hfi_shock_policy_change.png", width: 92%),
  caption: [Impulse responses to a 1 percentage point contractionary rate-change-only HFI shock, BVAR estimated through 2022.],
) <fig:hfi-irfs>

=== Subsample Comparison

Subsample estimation compares the benchmark-rate era (2002–2015) with the FR007-corridor era before the counterfactual window (2015–2022). The point estimates remain qualitatively similar across regimes, but the later estimates have substantially wider confidence bands because the shorter post-2015 window leaves less effective information for a 6-lag monthly specification. For that reason, the full-sample estimation remains the preferred baseline for precision, while the subsample evidence serves as a robustness check on whether the transmission patterns used for the 2023–2025 counterfactual are also visible in the modern operating regime. The narrative subsample figures appear in @fig:narrative-irfs-seg-2002-2015 and @fig:narrative-irfs-seg-2015-2022; the HFI counterparts appear in @fig:hfi-irfs-seg-2002-2015 and @fig:hfi-irfs-seg-2015-2022.


== Baseline Forecast

The counterfactual requires a no-policy-change baseline, the BVAR's iterated forecast from the end of the estimation sample. We estimate the BVAR through 2022 and generate recursive forecasts for 2023-2025.

GDP and IP forecasts track realized values well. The FR007 forecast stabilizes around 2.5%, substantially above the declining realized path toward approximately 1.5% by end-2025, reflecting the BVAR's inability to anticipate the post-2022 easing cycle, which is precisely the policy variation the counterfactual is designed to evaluate. The REER forecast exhibits a similar upward drift relative to realized values.

The CPI forecast merits particular attention. The BVAR projects CPI of approximately 1.75% over 2023-2025, against realized values near zero. Even when the BVAR is re-estimated through 2025, the out-of-sample CPI forecast settles around 1%, still well above realized inflation (Appendix Figure @fig:bvar-cpi-forecast-2025). This persistent overprediction is not surprising, the BVAR is estimated on a sample in which CPI rarely approached zero and never sustained a prolonged period of near-deflation, but it has a direct consequence for the counterfactual. Because the baseline already expects CPI well above zero, measuring the counterfactual gap against the baseline would understate the policy challenge: the relevant question is not how much additional inflation the counterfactual generates relative to an already-optimistic baseline, but how close the counterfactual brings CPI to the government's stated target. This motivates the evaluation against GWR targets rather than against the BVAR baseline.

#figure(
  image("outputs/diagnostics/2022/bvar_gdp_forecast.png", width: 92%),
  caption: [BVAR GDP forecast vs. realized values, 2022 estimation sample, recursive out-of-sample 2023–2025.],
) <fig:bvar-gdp-forecast-2022>

#figure(
  image("outputs/diagnostics/2022/cpi_forecast_vs_actual_until_2026.png", width: 92%),
  caption: [BVAR CPI forecast vs. realized values, 2022 estimation sample, recursive out-of-sample 2023–2025.],
) <fig:bvar-cpi-forecast-2022>

== 5.3 Counterfactual <sec:counterfactual>

We present counterfactual results for three targeting specifications, ordered from the most to the least comprehensive in terms of the PBoC's objective set. The baseline throughout is FR007 smoothing ($lambda_i = 1$), which penalizes abrupt rate changes and prevents the optimizer from producing implausible rate paths. 

=== Full Specification: CPI + GDP + REER

The full specification ($lambda_pi = 1, lambda_y = 1, lambda_i = 1, lambda_e = 1$) is the most policy-relevant scenario, as it approximates the PBoC's revealed multi-objective mandate: price stability, output growth, and exchange rate management. Under this specification:

- GDP tracks slightly above the BVAR forecast and realized values.
- FR007 and the REER largely follow the baseline forecast: FR007 increases relative to the realized path, and the REER appreciates modestly above realized values for some periods. Because the FR007 deviation from the baseline is small, IP is essentially unaffected.
- CPI rises only slightly above the BVAR forecast, settling around 2% under the 2022 sample, and around 1.3% under the 2025 sample.

This is the central finding of the paper. When the counterfactual optimizer is tasked with simultaneously closing the CPI gap, stabilizing GDP near target, and limiting exchange rate volatility, the target shortfall remains large. The implication is not simply that policy makers chose too little easing, but that three forces interact: the exchange-rate objective materially constrains the feasible policy path, conventional short-rate transmission is too weak to close the inflation gap cleanly, and the post-COVID economy may no longer obey the inflation dynamics embedded in the pre-2023 estimation sample. Conditional on the estimated transmission mechanism and the multi-objective loss function, the realized policy path therefore appears closer to the constrained optimum than a simple target-miss narrative would suggest.

#figure(
  image("outputs/main_results/2022/cnfctl_scenario_cpi_plus_gdp_plus_neer_2023_s2022.png", width: 100%),
  caption: [Counterfactual paths under the full specification ($lambda_pi = lambda_y = lambda_i = lambda_e = 1$, CPI + GDP + REER targeting), BVAR estimated through 2022. Shaded band: 68% posterior interval. Dashed: BVAR baseline forecast. Dotted red: GWR target.],
) <fig:cf-full-2022>

=== CPI + GDP

Relaxing the exchange rate objective ($lambda_e = 0$) while retaining the GDP target reveals the tradeoff structure. GDP now tracks close to target and to realized values, which is consistent with the PBoC's effective prioritization of growth. CPI becomes more smoothed, falling slightly below the GWR target for some periods. The key difference appears in FR007: the counterfactual rate path rises to 3–3.5%, substantially above both the BVAR forecast and realized values. This generates a correspondingly larger REER appreciation and a decline in IP growth from the 5% path to approximately 3%.

The economic interpretation is straightforward: without the exchange rate constraint, the optimizer exploits the CPI price puzzle, that is higher rates raise CPI through the housing cost channel in the short run, but the cost is borne by the external sector. The fact that removing the FX objective produces such different rate and exchange rate paths underscores the extent to which the exchange rate constraint binds in the full specification.

#figure(
  image("outputs/main_results/2022/cnfctl_scenario_cpi_plus_gdp_targeting_2023_s2022.png", width: 100%),
  caption: [Counterfactual paths under CPI + GDP targeting ($lambda_pi = lambda_y = lambda_i = 1, lambda_e = 0$), BVAR estimated through 2022. Shaded band: 68% posterior interval. Dashed: BVAR baseline forecast. Dotted red: GWR target.],
) <fig:cf-cpi-gdp-2022>

=== CPI Only

The most restrictive specification ($lambda_y = 0, lambda_e = 0$) targets only CPI deviation from the GWR target, subject to rate smoothing. Here, FR007 needs to be lower than the BVAR forecast (which projects rates above 2.25%) but remains higher than the realized path. GDP overshoots to approximately 7%. The REER appreciates above the baseline. IP remains close to both the forecast and the actual path, as the FR007 deviation from the baseline forecast is relatively small under this specification, the transmission into IP is muted.

This specification isolates the partial-equilibrium effect of CPI targeting: the optimizer generates only a modest deviation from the baseline rate path and keeps CPI somewhat closer to target, but without accounting for the output and exchange-rate consequences. The fact that even pure CPI targeting does not deliver a clean closure of the target gap reinforces the finding from the full specification: the problem is not simply insufficient responsiveness to inflation, but the limited capacity of the estimated rate-transmission mechanism to implement the target in the first place.

#figure(
  image("outputs/main_results/2022/cnfctl_scenario_cpi_targeting_2023_s2022.png", width: 100%),
  caption: [Counterfactual paths under strict CPI targeting ($lambda_pi = lambda_i = 1, lambda_y = lambda_e = 0$), BVAR estimated through 2022. Shaded band: 68% posterior interval. Dashed: BVAR baseline forecast. Dotted red: GWR target.],
) <fig:cf-cpi-only-2022>

=== Discussion

The progression from the full specification to the CPI-only case reveals a clear hierarchy of binding constraints. The exchange-rate constraint is the tightest: removing it changes the counterfactual rate path by 1-1.5pp and produces qualitatively different REER dynamics. The GDP constraint is effectively slack as growth was near target in 2023-2025, and the optimizer does not need to trade off CPI against output. The rate-smoothing penalty prevents all specifications from producing implausible rate volatility. Yet even this hierarchy does not exhaust the interpretation of the results: the weak capacity of FR007 movements to deliver target-consistent CPI paths indicates that the policy problem is partly one of impaired transmission, not merely one of objective weighting.

A post-2023 caveat applies to all three specifications. The BVAR baseline inherits the sample's central tendency, in which inflation averaged well above zero, and therefore forecasts CPI around 1.75% even as realized CPI collapsed. The counterfactual CPI paths are thus computed relative to an already-elevated baseline, which means their levels should be read with caution. Even under the full specification, the counterfactual CPI remains well below the GWR target. The counterfactual FR007 path stays near the BVAR forecast, which is substantially above the realized rate path, implying that the optimizer does not view the realized easing cycle as sufficient to close the CPI gap once exchange-rate and output objectives are internalized. The 2023-2025 inflation undershoot should therefore be read as the joint outcome of constrained policy optimization, weakened rate-based transmission, and a likely post-COVID shift in the inflation process itself.


== Robustness

=== Sensitivity to Loss Function Weights

The results are robust to raising the inflation weight. Doubling $lambda_pi$ from 1 to 2, placing greater emphasis on CPI stabilization relative to output and exchange rate objectives, yields counterfactual CPI and FR007 paths that are qualitatively indistinguishable from the baseline equal-weight specification. The binding constraint remains the exchange rate objective: the counterfactual rate path and the implied REER appreciation are driven by the presence of $lambda_e$, not by the precise value of $lambda_pi$. @fig:cf-lambda-sensitivity reports the full comparison.

=== CMPI-Based Counterfactuals

As an alternative transmission channel, @fig:cf-cmpi-2022 reports counterfactual paths computed using the Composite Monetary Policy Index (CMPI) shocks from #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose") in place of the joint FR007-based narrative--HFI transmission map. The CMPI aggregates changes across the 7-day reverse repo rate, the 1-year MLF, the 1-year LPR, the reserve requirement ratio, and pre-2015 benchmark rates into a single index, with each instrument converted to a common FR007-equivalent unit. Interpretation requires caution: the transmission maps here are built from CMPI IRFs rather than FR007 IRFs, and a unit CMPI shock does not carry the same policy meaning as a unit FR007 shock; the equal-weighting across instruments is atheoretical. As discussed in §A.4.3, the CMPI IRFs remove the initial positive responses of GDP and IP and shorten the CPI price puzzle relative to the FR007-based IRFs. Even so, the counterfactual CPI and FR007 paths are qualitatively similar to the FR007-based baseline, providing broad support for the robustness of the main findings. Differences in the magnitude of counterfactual FR007 deviations likely reflect the amplified IP response documented in the CMPI IRFs.

#figure(
  image("outputs/main_results/2022/cnfctl_cmpi_scenario_2023_s2022.png", width: 100%),
  caption: [Counterfactual paths using CMPI shocks as the transmission input (rows: CPI targeting, CPI + GDP, CPI + GDP + REER), BVAR estimated through 2022. Shaded band: 68% posterior interval. Dashed: BVAR baseline forecast. Dotted red: GWR target.],
) <fig:cf-cmpi-2022>

=== Full-Sample Estimation

 @fig:cf-scenario-2025 reports counterfactual paths for all three targeting specifications using the BVAR and IRFs estimated through the full sample (2002-2025) rather than through 2022. The primary difference from the main specification lies in the baseline forecast: the 2025-sample BVAR projects CPI of approximately 1% (versus ~1.75% with the 2022 sample), reflecting the realized disinflation that the 2022 model could not anticipate. The IRFs are qualitatively similar across sample endpoints, so the estimated transmission mechanism is broadly stable. Under the full specification (CPI + GDP + REER), counterfactual CPI settles around 1.3%, narrowing but not eliminating the gap relative to the GWR target. The qualitative interpretation is preserved: the inflation shortfall cannot be reduced to a simple failure of discretionary optimization, because even after incorporating the post-2022 data the counterfactual still reflects a binding exchange-rate tradeoff and limited inflation leverage from rate policy. The 2025-sample results mitigate, but do not remove, the structural-break concern.

#figure(
  image("outputs/main_results/2025/cnfctl_scenario_2023_s2025.png", width: 100%),
  caption: [Counterfactual paths under all three targeting specifications (rows: CPI targeting, CPI + GDP, CPI + GDP + REER), BVAR estimated through 2025. Shaded band: 68% posterior interval. Dashed: BVAR baseline forecast. Dotted red: GWR target.],
) <fig:cf-scenario-2025>

= Discussion and Limitations <sec:discussion>

The counterfactual framework employed in this paper rests on several maintained assumptions, each of which bounds the scope of our conclusions. We discuss five limitations in turn, noting where each connects to potential extensions.

== Linearity of Transmission and the Lucas Critique

The CMW framework assumes that impulse responses estimated under the prevailing policy regime remain valid for the counterfactual policy path, that is, agents do not revise their expectations or behavior in response to the alternative shock sequence. This assumption is defensible when the counterfactual represents a modest recalibration of existing instruments within the historical span of observed policy movements (§4.1.2). It becomes more tenuous as the counterfactual diverges from historical experience. Relatedly, the monthly aggregation of daily HFI shocks by summation maintains a linearity assumption: transmission is additive across within-month shocks. If transmission is state-dependent, sign-asymmetric, or size-nonlinear, as models with occasionally binding constraints or threshold effects would predict, the aggregated monthly shock may be a poor summary of the within-month policy impulse. Establishing the conditions under which the fixed-IRF assumption breaks down, for instance when a sufficiently persistent or large sequence of shocks leads agents to infer a regime change rather than a sequence of discretionary deviations, would be a valuable contribution in its own right, and we leave this for future work.

== Single-Instrument Identification in a Multi-Instrument Framework

The main specification treats FR007 as a sufficient statistic for the PBoC's policy stance, on the argument that all instruments, including OMOs, RRR, MLF, LPR, and formerly benchmark rates, ultimately transmit through the interbank market and are reflected in FR007 movements (§4.2.3). This argument is most convincing in the post-2015 era, when the 7-day reverse repo rate became the PBoC's explicit policy anchor and FR007 responded directly to OMO rate changes. In the pre-2015 period, when the PBoC operated primarily through quantity-based tools (RRR adjustments, credit quotas) and administratively set benchmark rates, the extent to which these actions transmitted into FR007 is less clear, because FR007 may have captured liquidity conditions without fully reflecting the real-economy transmission of quantity-based easing or tightening.

One natural response is to restrict the sample to 2015–2025, where the FR007-as-policy-anchor assumption is most defensible. However, the subsample IRF comparison (§A.2.2) reveals that transmission appears weaker, not stronger, in the post-2015 period, with wider credible bands and attenuated real-variable responses. This finding persists even under CMPI-based identification, which accounts for the full instrument toolkit. Two interpretations are possible. The first is statistical: the post-2015 sample is simply too short (roughly 120 monthly observations with 6 lags) to estimate a 7-variable BVAR with adequate precision. The second is substantive: monetary policy transmission may have genuinely weakened during this period, whether due to structural changes in the financial system, the disruption of the COVID shock, or the post-2020 property downturn that impaired the credit channel. Distinguishing between these explanations is beyond the scope of this paper but has direct implications for the reliability of the counterfactual over the 2023–2025 window.

== Information Effects and Fiscal-Monetary Coordination

The CMW framework requires that monetary policy shocks contain no private information about fundamentals (§4.1.2). The CMPI-based IRFs (§A.4.3) provide some reassurance on this point: the initial positive GDP spike that characterizes both the narrative and HFI FR007-based identifications disappears under CMPI identification, suggesting that the spike reflects an information or anticipation effect specific to FR007 movements rather than a feature of monetary transmission per se. To the extent that information effects are concentrated in the short-run response and attenuate at longer horizons, their contamination of the counterfactual may be limited because it depends primarily on the cumulative transmission profile.

A distinct but related concern arises from the institutional structure of Chinese macroeconomic policymaking. The State Council coordinates monetary and fiscal policy at the central level (Das and Song, 2022), meaning that PBoC rate decisions are frequently bundled with fiscal measures such as stimulus packages, tax adjustments, or local government bond issuance directives. This coordination creates a third information channel beyond the Nakamura and Steinsson (2018) private-fundamentals channel and the Bauer and Swanson (2023) response-to-news channel: PBoC actions may signal a broader political commitment to stabilization that encompasses fiscal as well as monetary tools. If agents respond to this signal rather than to the rate change per se, the estimated IRFs conflate monetary transmission with the fiscal response, and the counterfactual captures the combined effect of coordinated policy rather than monetary policy in isolation. Furthermore, the response of local governments, which account for approximately 80% of total government expenditure, to central policy signals is heterogeneous and difficult to model, adding a further layer of uncertainty. Formally disentangling monetary from fiscal transmission in the Chinese institutional context, for instance by classifying PBoC actions as coordinated versus uncoordinated and testing whether IRF patterns differ across the two regimes, remains an open question for future research.

== High-Frequency Identification: Data Limitations

The HFI shock series used in the joint transmission map (§4.3) relies on daily close-to-close FR007 changes, which is a coarse measurement window relative to the intraday data used in the US HFI literature. Two data limitations constrain improvement. First, PBoC policy announcements, particularly MLF operations, cluster around the 15th of each month, coinciding with NBS releases of major macroeconomic indicators. This mid-month co-timing means the daily FR007 change on announcement dates may reflect the market's joint reaction to the policy action and the macro data release rather than the policy surprise alone (§4.3.4). Addressing this contamination rigorously would require intraday (ideally hourly) IRS data on FR007, which is available only for recent years and only to interbank market participants. Second, a comprehensive economic calendar covering not only NBS macro releases but also fiscal policy announcements, institutional reforms, and other policy signals would be needed to control for non-monetary information arriving on the same dates. For earlier years in the sample, constructing such a calendar may require accessing PBoC and State Council archives directly.

== Baseline Forecast and the Post-2022 Structural Break

The BVAR baseline projects CPI of approximately 1.75% over 2023-2025, against realized values near zero, which is a persistent and substantial overprediction that is not resolved by re-estimating through 2025 (which still yields ~1%). GDP and IP forecasts, by contrast, track realized values well. This asymmetry, with accurate output forecasts alongside a dramatically wrong inflation forecast, suggests that the post-2022 CPI collapse is driven by factors outside the BVAR's information set rather than by a general model failure.

Several candidate explanations merit consideration. First, the BVAR excludes labor market variables due to data quality constraints (§A.1.2). If the Phillips curve relationship is operative in China, and post-COVID surveyed unemployment did rise substantially, the omission of labor market slack could account for the CPI overprediction: the model sees output recovering but cannot observe the labor market weakness that suppresses wage growth and, through it, inflation. Second, the accuracy of the output data itself is an open question. If GDP or industrial production data overstate real activity in the post-COVID period, a concern that has been raised in the broader literature on Chinese macroeconomic statistics, then the BVAR's inflation forecast is conditioned on an output path that is more favorable than reality, producing an upward-biased CPI projection. Third, and more fundamentally, the post-2022 period may represent a breakdown in the monetary transmission mechanism that the BVAR is designed to capture. Under the financing-through-money-creation view, broad money growth translates into inflation only to the extent that it finances new lending to productive real economic activity. In the post-COVID environment, successive reductions in policy rates and reserve requirements did not translate into proportionate increases in credit to the real economy, because household and corporate loan demand was depressed by the property downturn and balance-sheet repair, meaning that the quantity-side transmission channel on which the BVAR's M2 variable is predicated was substantially impaired. The BVAR, estimated on a sample in which rate cuts reliably generated credit expansion and inflationary pressure, projects CPI on the basis of a transmission mechanism that may no longer have been operative after 2022.

These considerations do not invalidate the counterfactual but they qualify the interpretation of its CPI paths. The counterfactual is constructed relative to the BVAR baseline and inherits whatever misspecification the baseline contains. This is the principal reason we evaluate the counterfactual against GWR targets rather than against the baseline (§4.5.4, §5.3.4).

= Conclusion <sec:conclusion>

This paper asks what the trajectory of Chinese inflation, output, and the exchange rate would have looked like during the 2023--2025 disinflationary episode had the PBoC followed explicit targeting rules tied to the Government Work Report benchmarks. Using the empirics-only framework of CMW (2026), we combine a BVAR baseline projection with a joint transmission map constructed from narrative FR007 shocks and high-frequency FR007 surprises to recover the policy-shock sequence that best implements alternative targeting objectives.

The main result is that the inflation gap does not disappear under counterfactual targeting. Even when the loss function prioritizes CPI stabilization, counterfactual inflation remains below the announced target, and once output and exchange-rate objectives are added the gap widens further. The full-specification exercise shows that the exchange-rate objective is the tightest immediate constraint on policy. But the findings are not well summarized as a policy-tradeoff story alone. The weak ability of conventional short-rate adjustments to generate target-consistent inflation paths points to obstacles in the transmission mechanism itself, while the persistent failure of the pre-2023 BVAR to describe realized post-COVID inflation suggests that the Chinese economy may have undergone a structural shift in the relation between rates, money, activity, and prices.

This interpretation has two implications. First, the recent inflation shortfall should not be read simply as evidence that the PBoC failed to ease enough. Conditional on the estimated transmission mechanism and the multi-objective mandate, realized policy appears closer to a constrained optimum than a literal target-miss narrative would suggest. Second, a move toward more explicit inflation targeting would not by itself solve the underlying problem if the conventional interest-rate channel has weakened or if post-COVID fundamentals have changed in ways not captured by historical dynamics. Future work should therefore focus less on whether the PBoC could have mechanically matched the GWR target with larger rate changes, and more on why the mapping from monetary instruments to inflation appears to have become so fragile in the first place.

#let appendix-numbering(..nums) = {
  let nums = nums.pos()
  if nums.len() == 2 {
    numbering("A", nums.at(1))
  } else if nums.len() >= 3 {
    numbering("A.1", nums.at(1), nums.at(2))
  }
}

#counter(heading).update((0,))
#heading(numbering: none)[Appendix]
#set heading(numbering: appendix-numbering)
#counter(figure.where(kind: image)).update(0)
#counter(figure.where(kind: table)).update(0)
#set figure(numbering: n => numbering("A.1", 1, n))

== Data Construction

=== Monthly GDP via Stock-Watson Distribution <app:monthly-gdp>

Quarterly real GDP is the only available frequency from the NBS, but quarterly data produces unreliable narrative shock identification. CXZ (2025), who use quarterly frequency, obtain GDP responses exceeding 2% to a one-standard-deviation policy shock, far outside the plausible range. MANR (2026) use monthly data estimated by the New York Fed (Groen, 2020), which employs a dynamic factor model on dozens of indicator series. However, that series ends around 2020 and is therefore not suitable for our 2023-2025 counterfactual window.

We adopt the #cite(<StockWatson2010>, form: "prose") distribution method, which allocates quarterly GDP to months using the adding-up identity $Q_T = q_(3T-2) + q_(3T-1) + q_(3T)$ and monthly indicator series that track the expenditure components. The method fits a state-space model with a Kalman smoother, ensuring that the monthly estimates sum exactly to the quarterly total reported by the NBS.

_Monthly indicators used for distribution:_

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header(
    [*Component*],
    [*Series*],
    [*Notes*],
    [*Source*]
  ),
  [Consumption],
  [Total retail sales of consumer goods (社会消费品零售总额)],
  [Absolute value],
  [NBS],
  [Investment],
  [Urban fixed asset investment (城镇固定资产投资)],
  [Backward-extended using overall FAI growth rate for pre-availability period. Captures local government capital spending, partially offsetting the central-only fiscal expenditure series below.],
  [NBS],
  [Government spending],
  [Central government fiscal expenditure (全国政府财政支出)],
  [Excludes local government expenditure and transfer payments],
  [NBS],
  [Net exports],
  [Merchandise trade balance (exports − imports)],
  [Recorded in USD; converted to CNY using monthly average CNY/USD spot rate from daily Wind data],
  [NBS],
  [Deflator],
  [Monthly CPI index],
  [Used to deflate constructed nominal monthly GDP to real terms],
  [NBS]
)

The implementation proceeds in six steps. First, all inputs are seasonally adjusted, because the Stock--Watson distribution is applied to seasonally adjusted series. Quarterly GDP is adjusted with quarter-specific factors from a centered four-quarter moving-average decomposition; monthly indicators are adjusted with month-specific factors from centered twelve-month moving averages, using an additive correction for series that can take negative values, such as the trade balance.

Second, the quarterly GDP series is decomposed into a smooth trend and a detrended component. Let $Q_T$ denote quarterly GDP and let $s_t$ be a smooth monthly trend obtained by cubic-spline interpolation of log quarterly GDP. The corresponding quarterly trend is

$ S_T = s_(3T - 2) + s_(3T - 1) + s_(3T), $

so the detrended quarterly series is $tilde(Q)_T = Q_T / S_T$. Each monthly indicator is detrended analogously by dividing it by its own smooth monthly trend.

Third, detrended monthly GDP is modeled as

$ tilde(q)_t = mu_t + u_t, quad mu_t = beta_0 + beta' x_t, quad u_t = rho u_(t - 1) + epsilon_t, $

where $x_t$ collects the detrended monthly indicators. The quarterly observation equation imposes the adding-up restriction. At the end of quarter $T$,

$ tilde(Q)_T = sum_(j = 0)^2 w_(T, j) tilde(q)_(3T - j), quad
  w_(T, j) = s_(3T - j) / S_T, $

so quarterly GDP is the trend-weighted sum of the three latent monthly observations within the quarter.

Fourth, the parameters $(beta, rho, sigma_epsilon)$ are estimated by Gaussian maximum likelihood. In the code, $rho$ is transformed to remain in $(-1, 1)$ and $sigma_epsilon$ is exponentiated to remain positive; optimization is performed on the unconstrained parameters.

Fifth, conditional on the estimated parameters, a Kalman filter updates the three latent monthly states whenever a quarter-end GDP observation becomes available, and a Rauch--Tung--Striebel smoother recovers the full monthly path using both past and future quarterly information.

Finally, the estimated seasonally adjusted monthly path is multiplied back by the quarterly seasonal factors and then rescaled within each quarter so that

$ q_(3T - 2) + q_(3T - 1) + q_(3T) = Q_T $

holds exactly for the original NBS quarterly GDP series. The reported monthly GDP growth series is then computed as the year-on-year percentage change in the distributed monthly level.

As an accounting check, the distributed monthly series aggregates exactly back to the official quarterly GDP series. Over 2002–2025, the quarterly sum of the constructed monthly series has correlation 1.000 with reported NBS quarterly real GDP, while the level RMSE is $2.43 times 10^(-11)$ and the corresponding RMSE for quarterly year-on-year growth is $1.61 times 10^(-14)$ percentage points, both numerically indistinguishable from zero. These statistics verify the temporal-aggregation constraint rather than independently validating the within-quarter monthly path; the substantive robustness question is therefore whether results are sensitive to the chosen monthly indicator set or to replacing the constructed GDP proxy with an alternative monthly output measure.

To assess how strongly the inferred monthly allocation depends on each indicator block, we re-estimate the distribution model for all 16 subsets of the four monthly indicators and compute each block's Shapley contribution to the absolute deviation from a no-indicator benchmark path. Figure @fig:gdp-indicator-shapley shows that consumption accounts for 69.0% of the information used to shape the within-quarter path, investment for 22.8%, government spending for 6.7%, and net exports for 1.5%. These shares describe the indicators' contribution to monthly timing, not their shares in expenditure-side GDP.

#figure(
  image("outputs/diagnostics/gdp_indicator_shapley_contributions.svg", width: 82%),
  caption: [Indicator contributions to the inferred within-quarter monthly GDP allocation.],
) <fig:gdp-indicator-shapley>

=== BVAR Variable Definitions <app:bvar-vars>

The 7-variable BVAR is estimated at monthly frequency with the following variable ordering:

$ y_t = (g_t, "IP"_t, pi_t, i_t, m_t, e_t, y_t^("US"))' $

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header(
    [*Symbol*],
    [*Variable*],
    [*Transformation*],
    [*Source*]
  ),
  [$g_t$],
  [Real GDP],
  [YoY growth (%)],
  [Constructed via Stock-Watson (§A.1.1)],
  [$"IP"_t$],
  [Industrial Value Added (above designated scale)],
  [YoY growth (%)],
  [NBS],
  [$pi_t$],
  [CPI],
  [YoY (%)],
  [NBS],
  [$i_t$],
  [FR007 (7-day interbank repo rate)],
  [Level (%)],
  [Wind],
  [$m_t$],
  [M2 money supply],
  [YoY growth (%)],
  [PBoC],
  [$e_t$],
  [RMB Real Broad Effective Exchange Rate],
  [YoY change (%)],
  [BIS],
  [$y_t^("US")$],
  [US Industrial Production],
  [YoY growth (%)],
  [FRED]
)

_Variable selection rationale._

_GDP and IVA both included._ GDP enters as the primary output measure for the counterfactual loss function; IVA enters as an independent monthly indicator of real activity that does not rely on the Stock-Watson construction. This is not double-counting because IVA is an NBS-reported series, while monthly GDP is a constructed proxy from expenditure components.

_Consumption, FAI, net exports excluded from BVAR._ These expenditure components are already embedded in the monthly GDP construction. Including them separately would create mechanical correlation between the GDP proxy and its own inputs.

_Labor market variables excluded._ The registered unemployment rate (城镇登记失业率) is notoriously smoothed and barely moved during the 2008–09 global financial crisis. The surveyed unemployment rate (城镇调查失业率) is available only from 2018. Other labor market indicators lack sufficient sample length. CMW (2026) include labor market variables in their US specification; these are simply not available for China at comparable quality.

_US Industrial Production._ Controls for the external demand environment. Preferred over a US output gap measure because it is directly observable, captures manufacturing-cycle spillovers to Chinese exports, and is standard in the China VAR literature.

_M2._ Enters as an endogenous propagation channel capturing the quantity dimension of Chinese monetary policy, which was the dominant transmission mechanism in the pre-2015 era of interest rate repression.

_BIS Real Effective Exchange Rate._ Preferred over the CFETS nominal effective rate because (i) BIS REER is available for the full 2002–2025 sample, (ii) it adjusts for inflation differentials, and (iii) CFETS index weights are only available from 2016.

_Year-on-year transformations._ Standard for monthly China macro work. Period-on-period rates would be better suited for structural papers isolating policy reaction timing but create inconsistency issues with the BVAR setup.

=== Government Work Report Targets <app:gwr-targets>

The GWR CPI and GDP growth targets define the counterfactual benchmarks ($pi^*_t$, $g^*_t$). Targets step in March at each NPC announcement, not January. When the GWR states a range (e.g., "7–7.5%"), we take the midpoint. When no new target is announced, including the 2020 GDP case, the previously operative target value is carried forward so the benchmark series remains continuous.

#figure(
  image("outputs/figures/quarterly_real_gdp_growth_vs_target.png", width: 84%),
  caption: [Official quarterly real GDP growth and the Government Work Report GDP growth target. The annual target is shown as a step series and takes effect from the March NPC announcement month.],
) <fig:gdp-target-series>

#figure(
  image("outputs/figures/monthly_cpi_vs_target.png", width: 84%),
  caption: [Monthly CPI inflation and the Government Work Report CPI target. The annual target is shown as a step series and takes effect from the March NPC announcement month.],
) <fig:cpi-target-series>

For the narrative policy rule (§4.2.2), the exchange rate gap uses only the CNY/USD bilateral rate rather than the full CFETS basket, because (i) the USD rate is unambiguously the most important reference point for PBoC FX policy and (ii) CFETS basket weights are available only from 2016.

=== CMPI Reconstruction

The Composite Monetary Policy Indicator of #cite(<MirandaAgrippinoNenovaRey2026>, form: "prose") aggregates deviations from trend across the PBoC's full instrument toolkit into a single index. We reconstruct the CMPI series from 2002 to 2025 using the 13 instrument categories documented in MANR Table A.1 (sourced from CMPI.xlsx via Wind). Because several categories contain multiple maturities or subtypes, the extracted panel contains 37 underlying monthly series. Tables @tab:cmpi-instruments-1–@tab:cmpi-instruments-3 report the English instrument names, the number of available monthly observations, and the observed sample span in the extracted CMPI panel.

#figure(
  table(
    columns: (2.2fr, 1.8fr, 0.7fr, 1.1fr),
    align: (left, left, center, center),
    table.header(
      [*Category*],
      [*Instrument*],
      [*Months*],
      [*Observed span*],
    ),
    [Repo rates], [14-day repo], [305], [2000-08–2025-12],
    [Repo rates], [21-day repo], [228], [2007-01–2025-12],
    [Repo rates], [28-day repo], [302], [2000-11–2025-12],
    [Repo rates], [91-day repo], [305], [2000-08–2025-12],
    [Repo rates], [182-day repo], [305], [2000-08–2025-12],
    [Repo rates], [364-day repo], [282], [2002-07–2025-12],
    [Reverse repo rates], [7-day reverse repo], [312], [2000-01–2025-12],
    [Reverse repo rates], [14-day reverse repo], [312], [2000-01–2025-12],
    [Reverse repo rates], [21-day reverse repo], [251], [2005-02–2025-12],
    [Reverse repo rates], [28-day reverse repo], [312], [2000-01–2025-12],
    [Reverse repo rates], [91-day reverse repo], [312], [2000-01–2025-12],
    [Treasury cash rates], [3-month Treasury cash], [229], [2006-12–2025-12],
    [Treasury cash rates], [6-month Treasury cash], [225], [2007-04–2025-12],
    [Treasury cash rates], [9-month Treasury cash], [190], [2010-03–2025-12],
  ),
  caption: [CMPI instrument panel, part 1: open-market and Treasury-cash instruments. “Months” denotes non-missing monthly observations in the reconstructed panel.],
) <tab:cmpi-instruments-1>

#figure(
  table(
    columns: (2.2fr, 1.8fr, 0.7fr, 1.1fr),
    align: (left, left, center, center),
    table.header(
      [*Category*],
      [*Instrument*],
      [*Months*],
      [*Observed span*],
    ),
    [Standing Lending Facility (SLF) rates], [Overnight SLF], [144], [2014-01–2025-12],
    [Standing Lending Facility (SLF) rates], [7-day SLF], [144], [2014-01–2025-12],
    [Standing Lending Facility (SLF) rates], [1-month SLF], [119], [2016-02–2025-12],
    [Short-term Liquidity Operations (SLO) rates], [SLO], [147], [2013-10–2025-12],
    [Short-term Liquidity Operations (SLO) rates], [Reverse SLO], [145], [2013-12–2025-12],
    [Central bank bill issuance rates], [3-month central bank bill], [281], [2002-08–2025-12],
    [Central bank bill issuance rates], [6-month central bank bill], [283], [2002-06–2025-12],
    [Central bank bill issuance rates], [1-year central bank bill], [282], [2002-07–2025-12],
    [Central bank bill issuance rates], [3-year central bank bill], [253], [2004-12–2025-12],
    [Benchmark time-deposit rates], [3-month benchmark time deposit], [312], [2000-01–2025-12],
    [Benchmark time-deposit rates], [1-year benchmark time deposit], [287], [2002-02–2025-12],
    [Financial institutions' deposit rates at the central bank], [Required reserves], [312], [2000-01–2025-12],
    [Financial institutions' deposit rates at the central bank], [Excess reserves], [312], [2000-01–2025-12],
  ),
  caption: [CMPI instrument panel, part 2: liquidity facilities, central bank bills, and deposit-rate instruments.],
) <tab:cmpi-instruments-2>

#figure(
  table(
    columns: (2.2fr, 1.8fr, 0.7fr, 1.1fr),
    align: (left, left, center, center),
    table.header(
      [*Category*],
      [*Instrument*],
      [*Months*],
      [*Observed span*],
    ),
    [Medium- and long-term benchmark lending rates], [1–3 year benchmark lending], [312], [2000-01–2025-12],
    [Medium- and long-term benchmark lending rates], [3–5 year benchmark lending], [312], [2000-01–2025-12],
    [Medium-term Lending Facility (MLF) rates], [3-month MLF], [136], [2014-09–2025-12],
    [Medium-term Lending Facility (MLF) rates], [6-month MLF], [127], [2015-06–2025-12],
    [Medium-term Lending Facility (MLF) rates], [1-year MLF], [119], [2016-02–2025-12],
    [Pledged Supplementary Lending (PSL) rates], [PSL], [136], [2014-09–2025-12],
    [Loan Prime Rates (LPR)], [1-year LPR], [147], [2013-10–2025-12],
    [Loan Prime Rates (LPR)], [5-year LPR], [77], [2019-08–2025-12],
    [Reserve requirement ratios (RRR)], [Large-bank RRR], [268], [2003-09–2025-12],
    [Reserve requirement ratios (RRR)], [Small-bank RRR], [268], [2003-09–2025-12],
  ),
  caption: [CMPI instrument panel, part 3: lending-rate and quantity instruments.],
) <tab:cmpi-instruments-3>

The CMPI is used only for robustness; the main counterfactual uses the joint FR007-based narrative--HFI transmission map (§4.2--§4.3). The reason is that the counterfactual requires shocks on a variable with transparent economic interpretation: a one-unit FR007 shock maps to a basis-point interbank rate movement, while a one-unit CMPI shock is a weighted average across heterogeneous instruments with no direct policy interpretation.

#line(length: 100%)

#counter(figure.where(kind: image)).update(0)
#counter(figure.where(kind: table)).update(0)
#set figure(numbering: n => numbering("A.1", 2, n))

== Narrative Shock Identification: Supplementary Results

=== Policy Rule Regression Results

Table @tab:fr007-policy-rule reports the estimated coefficients from the FR007 policy rule (§4.2.2) for three samples: the full sample (2002–2022), the benchmark-rate era (2002–2015), and the FR007-corridor era (2015–2022). In the main 2002–2022 specification, the negative GDP-gap coefficient is positive and statistically significant while the positive GDP-gap coefficient is not significant at the 5% level, matching the asymmetry emphasized by MANR (2026). The same asymmetry reappears in the 2015–2022 corridor-era subsample, whereas the earlier 2002–2015 estimates are less precise.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header(
      [*Variable*],
      [*2002–2022*],
      [*2002–2015*],
      [*2015–2022*],
    ),
    [$"FR007"_(t - 1)$], [0.778], [0.729], [0.816],
    [], [(0.040)], [(0.055)], [(0.046)],
    [], [p < 0.001], [p < 0.001], [p < 0.001],
    [$"cpi_gap"_(t - 1)$], [0.045], [0.064], [−0.018],
    [], [(0.019)], [(0.027)], [(0.029)],
    [], [p = 0.017], [p = 0.019], [p = 0.538],
    [$"gap_pos"_(t - 1)$], [−0.024], [−0.051], [−0.015],
    [], [(0.015)], [(0.027)], [(0.012)],
    [], [p = 0.098], [p = 0.064], [p = 0.216],
    [$"gap_neg"_(t - 1)$], [0.051], [0.156], [0.025],
    [], [(0.019)], [(0.114)], [(0.011)],
    [], [p = 0.009], [p = 0.173], [p = 0.030],
    [$"fx_gap"_(t - 1)$], [8.988], [61.057], [−3.222],
    [], [(21.498)], [(56.012)], [(11.065)],
    [], [p = 0.676], [p = 0.277], [p = 0.772],
    [*Observations*], [252], [168], [96],
    [$R^2$], [0.718], [0.707], [0.829],
    [*Adjusted* $R^2$], [0.712], [0.698], [0.820],
  ),
  caption: [FR007 policy-rule regressions. Standard errors are reported in parentheses; p-values appear beneath each coefficient block.]
) <tab:fr007-policy-rule>

=== FR007 Responses to Pre-2015 Policy Actions

Table @tab:fr007-pre2015-policy-response reports the monthly response of FR007 in months with pre-2015 policy adjustments, supporting the use of FR007 as a market-based policy indicator even before the reverse-repo era.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header(
      [*Policy action*],
      [*Event months*],
      [*Corr. with Δ FR007*],
      [*Expected sign*],
      [*Non-event mean*],
    ),
    [RRR adjustments],
    [36],
    [$0.371$],
    [66.7%],
    [$-0.030$],
    [Benchmark rate adjustments],
    [22],
    [$0.171$],
    [54.5%],
    [$0.016$],
  ),
  caption: [Monthly FR007 response in pre-2015 policy-adjustment months, 2002–2014.]
) <tab:fr007-pre2015-policy-response>

=== Subsample IRF Comparison

Figures @fig:narrative-irfs-seg-2002-2015 and @fig:narrative-irfs-seg-2015-2022 report narrative-shock IRFs estimated on the 2002–2015 and 2015–2022 subsamples. The earlier window corresponds to the benchmark-rate era; the later window isolates the FR007-corridor era before the counterfactual begins. The later estimates are somewhat noisier, especially at longer horizons, but they provide the more relevant recent-regime comparison.

#figure(
  image("outputs/diagnostics/seg_2002_2015/irf_bvar_iv_svar.png", width: 92%),
  caption: [Narrative-shock impulse responses, 2002–2015 subsample.],
) <fig:narrative-irfs-seg-2002-2015>

#figure(
  image("outputs/diagnostics/seg_2015_2022/irf_bvar_iv_svar.png", width: 92%),
  caption: [Narrative-shock impulse responses, 2015–2022 subsample.],
) <fig:narrative-irfs-seg-2015-2022>

=== Sign Asymmetry

@fig:narrative-signsplit-pos and @fig:narrative-signsplit-neg present IRFs estimated separately for positive (contractionary) and negative (expansionary) narrative shocks. The shapes are approximately mirror images, consistent with the linearity assumption maintained by the CMW framework (§4.1.2). A visual comparison suggests approximate symmetry, though formal Wald-type testing would be required to confirm this statistically. The shock sequence here is identified via the first-difference policy rule (without the lagged rate term), normalised to a $+1$ pp FR007 move; the expansionary panel sign-flips the negative-shock responses to make them directly comparable.

#figure(
  image("outputs/diagnostics/2025/rr_irf_dfr007_nolag_contractionary_pos_stability_filtered_fx_current.png", width: 92%),
  caption: [Narrative sign-split IRFs: contractionary shocks (positive $Delta$FR007 residual, normalised to $+1$ pp). Full sample 2002–2025. Bands are 68% and 90% posterior credible sets.],
) <fig:narrative-signsplit-pos>

#figure(
  image("outputs/diagnostics/2025/rr_irf_dfr007_nolag_expansionary_neg_stability_filtered_fx_current.png", width: 92%),
  caption: [Narrative sign-split IRFs: expansionary shocks (negative $Delta$FR007 residual, sign-flipped to $+1$ pp equivalent). Full sample 2002–2025.],
) <fig:narrative-signsplit-neg>

#line(length: 100%)

#counter(figure.where(kind: image)).update(0)
#counter(figure.where(kind: table)).update(0)
#set figure(numbering: n => numbering("A.1", 3, n))

== HFI Shock Identification: Supplementary Results

=== Rate-Change-Only vs. All-Announcement Shocks

Figure A.X compares IRFs using the rate-change-only HFI series against the all-announcement series (which includes dates where the PBoC confirmed no change). The rate-change-only series produces substantially cleaner IRFs because the all-announcement series introduces noise from no-change dates where the close-to-close FR007 movement reflects liquidity conditions rather than policy surprises. This motivates our choice of the rate-change-only series as the preferred HFI input to the joint transmission map.

`[Figure: IRFs from rate-change-only vs. all-announcement HFI, all BVAR variables.]`

=== Policy-Event / Macro-Release Overlap

The Wind economic calendar files available for this project contain usable Chinese macro-release observations only from January 25, 2007 onward, so this exercise is descriptive rather than a full-sample correction. Table @tab:hfi-policy-macro-overlap compares same-day overlap rates for the broad all-announcement series and the preferred rate-change-only series.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header(
      [*Policy-event definition*],
      [*Calendar-covered dates*],
      [*Same-day overlaps*],
      [*Overlap share*],
    ),
    [All policy announcements], [3,714], [327], [8.8%],
    [Rate-change-only announcements], [167], [14], [8.4%],
  ),
  caption: [Same-day overlap between PBoC policy-event dates and major Chinese macro releases among calendar-covered observations from January 25, 2007 onward. Major releases include CPI, PPI, industrial value added, retail sales, fixed-asset investment, and GDP.],
) <tab:hfi-policy-macro-overlap>

=== HFI Subsample Comparison

Figures @fig:hfi-irfs-seg-2002-2015 and @fig:hfi-irfs-seg-2015-2022 report HFI-based IRFs for the same two windows. As with the narrative identification, the post-2015 estimates are noisier than the longer early-sample estimates; moreover, the rate-change-only first stage is weak in the shorter recent window, so this evidence is suggestive rather than decisive. The comparison is still useful because it shows how much of the full-sample HFI pattern survives when attention is restricted to the modern operating framework.

#figure(
  image("outputs/diagnostics/seg_2002_2015/irf_hfi_shock_policy_change.png", width: 92%),
  caption: [Rate-change-only HFI impulse responses, 2002–2015 subsample.],
) <fig:hfi-irfs-seg-2002-2015>

#figure(
  image("outputs/diagnostics/seg_2015_2022/irf_hfi_shock_policy_change.png", width: 92%),
  caption: [Rate-change-only HFI impulse responses, 2015–2022 subsample.],
) <fig:hfi-irfs-seg-2015-2022>

=== HFI Sign Asymmetry

@fig:hfi-signsplit-pos and @fig:hfi-signsplit-neg present rate-change-only HFI IRFs estimated separately for positive (contractionary) and negative (expansionary) shocks. The results are noisier than the narrative sign-split exercise (§A.2.3), because splitting the sample roughly halves the effective instrument variation, but the broad shapes are qualitatively consistent with approximate symmetry. The expansionary panel sign-flips the negative-shock responses to a $+1$ pp contractionary equivalent for direct comparison.

#figure(
  image("outputs/diagnostics/2025/irf_hfi_ratechange_signsplit_contractionary.png", width: 92%),
  caption: [HFI sign-split IRFs: contractionary shocks (positive rate-change-only HFI, normalised to $+1$ pp FR007). Full sample 2002–2025. Bands are 68% posterior credible sets.],
) <fig:hfi-signsplit-pos>

#figure(
  image("outputs/diagnostics/2025/irf_hfi_ratechange_signsplit_expansionary.png", width: 92%),
  caption: [HFI sign-split IRFs: expansionary shocks (negative rate-change-only HFI, sign-flipped to $+1$ pp equivalent). Full sample 2002–2025.],
) <fig:hfi-signsplit-neg>

=== HFI Shock Series

Figure A.X plots the monthly-aggregated rate-change-only HFI shock series over 2002–2025. Major episodes, including the 2008 easing, 2010–11 tightening, 2015 easing, and 2022–2024 easing, are visible as clusters of same-sign shocks.

`[Figure: Time series of monthly HFI shocks.]`

#line(length: 100%)

#counter(figure.where(kind: image)).update(0)
#counter(figure.where(kind: table)).update(0)
#set figure(numbering: n => numbering("A.1", 4, n))

== CMPI-Based Identification: Supplementary Results

=== CMPI Policy Rule Regression

Table A.X reports the estimated coefficients from the CMPI policy rule (replacing FR007 with CMPI as the dependent variable). The regression is reported for three samples as in §A.2.1.

`[Figure: CMPI regression table for the full sample, 2002–2015, and 2015–2022.]`

=== CMPI Shock Series

Figure @fig:cmpi-vs-fr007-narrative-shocks plots the CMPI narrative shock series alongside the FR007 narrative shock series for comparison. The two series are only weakly positively correlated, reflecting the CMPI's inclusion of non-rate instruments.

#figure(
  image("outputs/figures/cmpi_vs_fr007_narrative_shock_series.png", width: 92%),
  caption: [Narrative monetary policy shock series identified on FR007 and on the CMPI, 2002–2025.],
) <fig:cmpi-vs-fr007-narrative-shocks>

=== CMPI IRFs <app:cmpi-irfs>

Figure A.X reports IRFs to a 1pp CMPI shock for the 2002–2022 sample. The key differences from the FR007-based IRFs (§5.1.1) are:

- _GDP:_ No initial positive spike; slightly negative in the first period, reaching a trough of approximately −1.5pp around the eighth month before recovering. The absence of the initial spike is consistent with the CMPI capturing non-rate instruments that do not generate the same information-effect or monthly-frequency GDP-hike dynamics.
- _CPI:_ A much smaller initial increase (peak ~0.2pp), crossing zero by the fifth month and reaching a trough of −0.8pp around the twelfth month. This substantially reduced price puzzle corroborates the user-cost housing channel interpretation: the CMPI includes non-rate instruments (RRR, structural tools) that do not directly affect mortgage costs, attenuating the mechanical positive CPI response.
- _FR007:_ Very wide credible bands, which is expected because FR007 is only one of the instruments entering the CMPI, and a composite shock need not move the interbank rate consistently.
- _REER:_ Appreciates to approximately 2pp around the eighth month, qualitatively similar to the FR007-based response.
- _IP:_ Starts negative, reaches a trough of approximately −3pp around the eighth month, roughly triple the FR007-based IP response. This amplification may reflect the CMPI's inclusion of RRR changes, which directly affect bank lending capacity and thus investment-heavy industrial output.

The overall pattern suggests that accounting for the full instrument toolkit produces more conventional-looking IRFs, particularly for CPI. The equal-weight assumption remains a caveat (§4.2.4).

=== CMPI Sign Asymmetry

@fig:cmpi-signsplit-pos and @fig:cmpi-signsplit-neg present CMPI-based IRFs estimated separately for contractionary and expansionary shocks. The BVAR here includes both CMPI and FR007 as endogenous variables (six variables total), and the instrument is the CMPI policy-rule residual. As with the HFI sign-split, splitting the sample approximately halves the effective instrument variation and widens the credible sets, but the qualitative shapes are broadly symmetric, consistent with the linearity assumption in §4.1.2.

#figure(
  image("outputs/diagnostics/2025/irf_bvar_iv_svar_signsplit_contractionary_cmpi.png", width: 92%),
  caption: [CMPI sign-split IRFs: contractionary shocks (positive CMPI narrative residual, normalised to $+1$ CMPI unit). Full sample 2002–2025. Bands are 68% posterior credible sets.],
) <fig:cmpi-signsplit-pos>

#figure(
  image("outputs/diagnostics/2025/irf_bvar_iv_svar_signsplit_expansionary_cmpi.png", width: 92%),
  caption: [CMPI sign-split IRFs: expansionary shocks (negative CMPI narrative residual, sign-flipped). Full sample 2002–2025.],
) <fig:cmpi-signsplit-neg>

#line(length: 100%)

#counter(figure.where(kind: image)).update(0)
#counter(figure.where(kind: table)).update(0)
#set figure(numbering: n => numbering("A.1", 5, n))

== Baseline Forecast Diagnostics

=== BVAR Residuals and Forecast Decay

Figure A.X plots the BVAR residuals for each equation over the estimation sample. Figure A.Y plots the forecast root mean squared error as a function of forecast horizon (1 to 36 months) for each variable, illustrating the rate at which forecast precision deteriorates.

#figure(
  image("outputs/diagnostics/2022/bvar_residuals.png", width: 100%),
  caption: [BVAR residuals for each equation over the estimation sample (2002–2022). Each panel shows the posterior mean residual series for one variable; shaded band is the 68% posterior interval.],
) <fig:bvar-residuals-2022>

#figure(
  image("outputs/diagnostics/2022/bvar_wold_decay.png", width: 100%),
  caption: [Forecast RMSE by horizon (1–36 months) for each BVAR variable, 2022-sample estimation. Each panel reports the root mean squared forecast error from the Wold decomposition as the horizon extends, illustrating the rate at which predictability decays.],
) <fig:bvar-wold-decay-2022>

#counter(figure.where(kind: image)).update(0)
#counter(figure.where(kind: table)).update(0)
#set figure(numbering: n => numbering("A.1", 6, n))

== Counterfactual: Supplementary Results

=== Full-Sample Estimation (2025 Sample)

See Figure @fig:cf-scenario-2025 in §5.4.2. For reference, Figure @fig:cf-cmpi-2025 additionally reports the CMPI-based counterfactual using the 2025-sample BVAR.

#figure(
  image("outputs/main_results/2025/cnfctl_cmpi_scenario_2023_s2025.png", width: 100%),
  caption: [CMPI-based counterfactual paths under all three targeting specifications, BVAR estimated through 2025. Shaded band: 68% posterior interval. Dashed: BVAR baseline forecast. Dotted red: GWR target.],
) <fig:cf-cmpi-2025>

#figure(
  image("outputs/diagnostics/2025/bvar_gdp_forecast.png", width: 100%),
  caption: [BVAR out-of-sample GDP forecast, 2025-sample estimation. Shaded band: 68% posterior interval. The 2025-sample BVAR incorporates the post-2022 easing cycle, bringing the baseline GDP forecast closer to the realized path relative to the 2022-sample specification.],
) <fig:bvar-gdp-forecast-2025>

#figure(
  image("outputs/diagnostics/2025/cpi_forecast_vs_actual_until_2026.png", width: 100%),
  caption: [BVAR CPI forecast vs. realized values, 2025 estimation sample. Even after incorporating observations through 2025, the out-of-sample forecast remains above realized inflation.],
) <fig:bvar-cpi-forecast-2025>

=== Robustness to Replacing Monthly GDP with IVA

=== CMPI-Based Counterfactual

See Figure @fig:cf-cmpi-2022 in §5.4.1 for the main CMPI-based counterfactual (2022-sample BVAR) and Figure @fig:cf-cmpi-2025 above for the 2025-sample version.

=== Sensitivity to Loss Function Weights <app:sensitivity-lambda>

Figure @fig:cf-lambda-sensitivity reports counterfactual paths under the full specification ($lambda_pi = lambda_y = lambda_i = lambda_e = 1$, top row) against a more inflation-focused variant ($lambda_pi = 2, lambda_y = lambda_i = lambda_e = 1$, bottom row), both using the BVAR estimated through 2022. Doubling the inflation weight shifts the loss function toward tighter CPI stabilization, but the counterfactual CPI and FR007 paths are qualitatively indistinguishable from the baseline. The exchange rate objective remains the binding constraint in both specifications: the key features of the counterfactual, namely a rate path above realized FR007 and the associated REER appreciation, are preserved. This confirms that the main results are not sensitive to the precise value of $lambda_pi$ within a plausible range.

#figure(
  image("outputs/main_results/2022/cnfctl_scenario_compare_focus_2023_s2022.png", width: 100%),
  caption: [Sensitivity to inflation weight: full specification ($lambda_pi = lambda_y = lambda_i = lambda_e = 1$, top) vs.\ more CPI-focused specification ($lambda_pi = 2$, bottom), BVAR estimated through 2022. Shaded band: 68% posterior interval. Dashed: BVAR baseline forecast. Dotted red: GWR target.],
) <fig:cf-lambda-sensitivity>

#pagebreak()

// ============================================================================
// References
// ============================================================================

#bibliography("reference.bib", title: "References", style: "chicago-author-date")
