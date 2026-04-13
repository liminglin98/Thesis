# =============================================================================
# RUN ALL — Full pipeline for BVAR / IV-SVAR / Counterfactual analysis
# =============================================================================
#
# Usage:  julia src/julia/run_all.jl
#
# Execution order:
#   1. LTP.jl              — BVAR with Minnesota prior (7-var, 120-month horizon)
#   2. RRShocks_monthly.jl — Narrative policy shocks + BVAR+IV-SVAR
#   3. HFIShocks.jl        — HFI shocks + BVAR+IV-SVAR
#   4. Counterfactual.jl   — Counterfactual scenarios (depends on 1–3)
#
# Each of scripts 1–3 loops over 2 sample periods:
#   2025 (baseline):  2002-01 to 2025-12
#   2022 (pre-deflation): 2002-01 to 2022-12
#
# Outputs are saved to (with year subfolder per sample):
#   outputs/intermediate/{2025,2022}/  — serialized .jls data files
#   outputs/main_results/{2025,2022}/  — IRF and counterfactual plots
#   outputs/diagnostics/{2025,2022}/   — BVAR diagnostic plots
#   outputs/robustness/{2025,2022}/    — robustness check plots
# =============================================================================

using Dates, Printf

const JULIA_DIR = @__DIR__

println("="^70)
println("  FULL PIPELINE — BVAR / IV-SVAR / COUNTERFACTUAL")
println("  Started: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("="^70)

t_total = time()

scripts = [
    ("LTP.jl",              "BVAR with Minnesota Prior"),
    ("RRShocks_monthly.jl", "Narrative Shocks + IV-SVAR"),
    ("HFIShocks.jl",        "HFI Shocks + IV-SVAR"),
    ("Counterfactual.jl",   "Counterfactual Scenarios"),
]

timings = Float64[]

for (i, (script, desc)) in enumerate(scripts)
    println("\n", "#"^70)
    println(@sprintf("# [%d/%d] %s  (%s)", i, length(scripts), desc, script))
    println("#"^70, "\n")

    t_start = time()
    include(joinpath(JULIA_DIR, script))
    elapsed = time() - t_start
    push!(timings, elapsed)

    println(@sprintf("\n>> %s finished in %.1f seconds", script, elapsed))
end

total_elapsed = time() - t_total

println("\n", "="^70)
println("  PIPELINE COMPLETE")
println("="^70)
println(@sprintf("  %-30s  %8s", "Script", "Time (s)"))
println("  ", "-"^40)
for (i, (script, _)) in enumerate(scripts)
    println(@sprintf("  %-30s  %8.1f", script, timings[i]))
end
println("  ", "-"^40)
println(@sprintf("  %-30s  %8.1f", "Total", total_elapsed))
println()
println("  Output directories (per sample: 2025, 2022):")
println("    intermediate/{year}/   — .jls data files")
println("    main_results/{year}/   — IRF & counterfactual plots")
println("    diagnostics/{year}/    — BVAR diagnostic plots")
println("    robustness/{year}/     — robustness check plots")
println("="^70)
