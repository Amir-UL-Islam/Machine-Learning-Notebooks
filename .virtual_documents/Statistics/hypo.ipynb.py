import numpy as np
import pandas as pd

# Reading data
late_shipments = pd.read_feather('data/late_shipments.feather', columns=None, use_threads=True)

# Print the late_shipments dataset
display(late_shipments.head())

# Calculate the proportion of late shipments
late_prop_samp = (late_shipments['late']=='Yes').mean()

# Print the results
# print(late_prop_samp)
# print(late_shipments)




def bootstrap(x, nboot, operation):
    """This function will return a bootstrap sampling distribution
    
    Args:
        x(list):  a list.
        nboot(int): number of bootstrap samples
        operation: which operation will be executed on the sample.
    
    Return:
        list: late_shipments_boot_distn
        
    """
    # making a numpy array from x, so that we can use the x[index]. This process will allow us
    # to take sample with replecement.
    x = np.array(x)
    
    late_shipments_boot_distn = []
    for i in range(nboot):
        index = np.random.randint(0, len(x), len(x))
        samples = x[index]
        late_shipments_boot_distn.append(operation(samples))
        
    return np.array(late_shipments_boot_distn)

    


# Calling the bootstrap function and assign the value to *late_shipments_boot_distn
late_shipments_boot_distn = bootstrap(late_shipments['late']=='Yes', 5000, np.mean)

# Hypothesize that the proportion is 6%
late_prop_hyp = 0.06

# Calculate the standard error
std_error = np.std(late_shipments_boot_distn, ddof=1)

# Find z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Print z_score
print(z_score)


from scipy.stats import norm

# Calculate the z-score of late_prop_samp
z_score = (late_prop_samp-late_prop_hyp)/std_error

# Calculate the p-value
p_value = 1-norm.cdf(z_score)
                 
# Print the p-value
print(p_value) 


# Calculate 95% confidence interval using quantile method
lower = np.quantile(late_shipments_boot_distn, 0.05)
upper = np.quantile(late_shipments_boot_distn, 0.95)

# Print the confidence interval
print((lower, upper))


# Reading the data
stack = pd.read_feather('data/stack_overflow.feather', columns=None, use_threads=True)
display(stack.head())


# Groupby Mean
xbar = stack.groupby('age_first_code_cut')['converted_comp'].mean()

# Groupby std
s = stack.groupby('age_first_code_cut')['converted_comp'].std()

# n count
n = stack.groupby('age_first_code_cut')['converted_comp'].count()



# Defining xbar_yes, no, s_yes ....
yes =late_shipments['late'] == 'Yes'
no = late_shipments['late'] == 'No'
xbar_yes = yes.mean()
s_yes = yes.std()
xbar_no = no.mean()
s_no = no.std()
n_yes = yes.count()
n_no = no.count()

# Calculate the numerator of the test statistic
numerator = xbar_yes - xbar_no

# Calculate the denominator of the test statistic
denominator = np.sqrt(s_yes**2/n_yes+s_no**2/n_no)

# Calculate the test statistic
t_stat = numerator/denominator

# Print the test statistic
print(t_stat)


from scipy.stats import t

x =1- t.cdf(t_stat, df=True)
print(x)






