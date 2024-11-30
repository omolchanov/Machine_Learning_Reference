# Implementation of Bayes theorem
# https://habr.com/ru/articles/598979/

def identify_drug_user(
        probability_threshold=0.5,
        prevalence=0.05,
        sensitivity=0.99,
        specificity=0.99):

    p_user = prevalence
    p_non_user = 1-p_user

    p_pos_user = sensitivity
    p_neg_user = specificity
    p_pos_non_user = 1-p_neg_user

    numerator = p_pos_user * p_user
    denumenator = p_pos_user * p_user + p_pos_non_user * p_non_user

    probability = round(numerator / denumenator, 3)

    print('\nProbability:', probability)

    if probability > probability_threshold:
        print('The user is probably drug addicted')

    else:
        print('The user is probably NOT drug addicted')

    return probability


identify_drug_user()

# Chain of aposteriori -> apriori probabilities, information gain

# Round 1
p1 = identify_drug_user()
print('Probability of the test-taker being a drug user, in the first round of test, is: ', p1)

# Round 2
p2 = identify_drug_user(prevalence=p1)
print('Probability of the test-taker being a drug user, in the second round of test, is: ', p2)

# Round 3
p3 = identify_drug_user(prevalence=p2)
print('Probability of the test-taker being a drug user, in the third round of test, is: ', p3)
