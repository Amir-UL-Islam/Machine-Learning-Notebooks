import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')




df_real_state = pd.read_csv('data/taiwan_real_estate2.csv')
df_real_state.head()


# Import ols from statsmodels.formula.api
from statsmodels.formula.api import ols

# Fit a linear regression of price_twd_msq vs. n_convenience
mdl_price_vs_conv = ols("price_twd_msq ~ n_convenience", data=df_real_state ).fit()

# Print the coefficients
print(mdl_price_vs_conv.params)


# Fit a linear regression of price_twd_msq vs. house_age_years, no intercept
mdl_price_vs_age = ols("price_twd_msq ~ house_age_years + 0", data=df_real_state).fit()
print(mdl_price_vs_age.params, '\n')



# Fit a linear regression of price_twd_msq vs. n_convenience plus house_age_years, no intercept
mdl_price_vs_both =  ols("price_twd_msq ~ n_convenience + house_age_years + 0", data=df_real_state).fit()

# Print the coefficients
print(mdl_price_vs_both.params)


fig =plt.figure(figsize=(12, 7))
# Create a scatter plot with linear trend line of price_twd_msq vs. n_convenience
sns.regplot(x='n_convenience',y='price_twd_msq', data=df_real_state)

# Show the plot
plt.show()



fig =plt.figure(figsize=(12, 7))
# Create a boxplot of price_twd_msq vs. house_age_years
sns.boxenplot(x='house_age_years',y='price_twd_msq', data=df_real_state)
plt.show()



# Extract the model coefficients, coeffs
coeffs = mdl_price_vs_both.params

# Print coeffs
print(coeffs)

# Assign each of the coeffs
ic_0_15, ic_15_30, ic_30_45, slope = coeffs


fig =plt.figure(figsize=(12, 7))

# Draw a scatter plot of price_twd_msq vs. n_convenience colored by house_age_years
sns.scatterplot(x='n_convenience', y='price_twd_msq', hue='house_age_years', data=df_real_state)

# Add three parallel lines for each category of house_age_years
# Color the line for ic_0_15 blue
plt.axline(xy1=(0, ic_0_15), slope=slope, color="blue")
# Color the line for ic_15_30 orange
plt.axline(xy1=(0, ic_15_30), slope=slope, color="orange")
# Color the line for ic_30_45 green
plt.axline(xy1=(0, ic_30_45), slope=slope, color="green")

# Show the plot
plt.show()
# Show the plot
plt.show()


df_fish = pd.read_csv('data/fish.csv')
df_fish.head()


# Fit a linear regression of species and length_cm to mass_g
length_vs_mass = ols("mass_g ~ length_cm", data=df_fish ).fit()

# Print the coefficients
print(length_vs_mass.params)



from itertools import product

length_fish = np.arange(5, 61, 5)
length_fish


species = df_fish['species'].unique()
species


product_of_species = product(species, length_fish)
product_of_species = [i for i in product_of_species]


extend_df_fish = pd.DataFrame(product_of_species, columns=['species','length_cm'])
extend_df_fish.head()


extend_df_length = extend_df_fish['length_cm']
extend_df_length


# Fit a linear regression of (species and length_cm) to mass_g
both_species_and_length_vs_mass_g = ols("mass_g ~ length_cm + species + 0 ", data = df_fish).fit()
both_species_and_length_vs_mass_g.params


predictions_of_length = extend_df_fish.assign(mass_g = both_species_and_length_vs_mass_g.predict(extend_df_fish))
predictions_of_length


# Fit a linear regression of (species and length_cm) to mass_g
species_length_vs_mass_both = ols("mass_g ~ length_cm + species + 0", data=df_fish).fit()

# Print the coefficients
print(species_length_vs_mass_both.params)
it_br, it_pe, it_pi, it_ro, slope = species_length_vs_mass_both.params


fig =plt.figure(figsize=(12, 7))

sns.scatterplot(x='length_cm',
                y='mass_g',
                hue='species',
                data=df_fish)
sns.scatterplot(x='length_cm',
                y='mass_g',
                color='black',
                data=predictions_of_length)

plt.axline(xy1=(0, it_br), slope=slope, color='blue')
plt.axline(xy1=(0, it_pe), slope=slope, color='green')
plt.axline(xy1=(0, it_pi), slope=slope, color='yellow')
plt.axline(xy1=(0, it_ro), slope=slope, color='red')
plt.show()


conditions = [
    predictions_of_length['species'] == 'Bream',
    predictions_of_length['species'] == 'Perch',
    predictions_of_length['species'] == 'Pike',
    predictions_of_length['species'] == 'Roach'
]
choices = [it_br, it_pe, it_pi, it_ro]


intercept = np.select(conditions, choices)


predictions_of_length_scores = extend_df_fish.assign(
    intercept = np.select(conditions, choices),
    mass_g = intercept + slope * extend_df_fish['length_cm']
)
predictions_of_length_scores.head()


predictions_of_length.head()



















































