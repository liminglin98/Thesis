##
using Pkg
using CSV, DataFrames
using Statistics
using GLM
using StatsModels
using Plots
using Dates
using ShiftedArrays, CovarianceMatrices
##
cd(@__DIR__)
pwd()
##
# Load the data
df = CSV.read("crz2025.csv", DataFrame);
##
# Create new variables for positive and negative GDP gaps
df.not_meet_target = df.gdp_gap .< 0;
df.gap_pos = df.gdp_gap .* (1 .- df.not_meet_target);
df.gap_neg = df.gdp_gap .* df.not_meet_target;
##
# Create lagged variables for the regression
df.FR007_l1   = lag(df.FR007, 1);
df.cpi_gap_l1 = lag(df.cpi_gap, 1);
df.gap_pos_l1 = lag(df.gap_pos, 1);
df.gap_neg_l1 = lag(df.gap_neg, 1);
##
# Drop missing values
df = dropmissing(df, [:FR007, :FR007_l1, :cpi_gap_l1, :gap_pos_l1, :gap_neg_l1])
## 
# Run the regression strictly followig CRZ2025
model1 = lm(@formula(FR007 ~ FR007_l1 + cpi_gap + gap_pos + gap_neg), df)
##
# Generate predictions from the model
df.pred_FR007 = predict(model1)
df.residuals = residuals(model1)
##
# Plot actual vs predicted values
plot(df.quarter, df.FR007, label="Actual FR007", legend=:topleft)
plot!(df.quarter, df.pred_FR007, label="Predicted FR007", linestyle=:dash)
xlabel!("Quarter")
ylabel!("FR007")
title!("Actual vs Predicted FR007 (CRZ2025)")
##
# Run the regression with lag 1 variables (MANR2026)
model2 = lm(@formula(FR007 ~ FR007_l1 + cpi_gap_l1 + gap_pos_l1 + gap_neg_l1), df)
##
# generate predictions from the model
df.pred_FR007_lag = predict(model2)
df.residuals_lag = residuals(model2)
##
# Plot actual vs predicted values for the lagged model
plot(df.quarter, df.FR007, label="Actual FR007", legend=:topleft)
plot!(df.quarter, df.pred_FR007_lag, label="Predicted FR007 (Lagged)", linestyle=:dash)
xlabel!("Quarter")
ylabel!("FR007")
title!("Actual vs Predicted FR007 (MANR2026)")