import pickle

from experiment_common import *

outcomes = {}

for k in n_training_images:
    size_dir = os.path.join(output_dir, 'size%i' % k)
    for sibling_weight in sibling_weights:
        sw_dir = os.path.join(size_dir,
            'sibling_weight_' + formatted_string(sibling_weight))
        if not os.path.exists(sw_dir):
            continue
        outcomes[sibling_weight] = {}
        for parent_weight in parent_weights:
            pw_dir = os.path.join(sw_dir,
                'parent_weight_' + formatted_string(parent_weight))
            if not os.path.exists(pw_dir):
                continue
            outcomes[sibling_weight][parent_weight] = {}
            for alpha in alphas:
                alpha_dir = os.path.join(pw_dir, 'alpha_' + formatted_string(alpha))
                if not os.path.exists(alpha_dir):
                    continue
                outcomes[sibling_weight][parent_weight][alpha] = {}
                for beta in betas:
                    beta_dir = os.path.join(alpha_dir, 'beta_' + formatted_string(beta))
                    outcome_file = os.path.join(beta_dir, 'outcomes.pkl')
                    if not os.path.exists(outcome_file):
                        continue
                    outcomes[sibling_weight][parent_weight][alpha][beta] = {}
                    local_outcomes = pickle.load(open(outcome_file))
                    outcomes[sibling_weight][parent_weight][alpha][beta]['intensity'] = \
                      sum(local_outcomes['itensity'])
                    outcomes[sibling_weight][parent_weight][alpha][beta]['gradient'] = \
                      sum(local_outcomes['gradient'])

text_values = ''
# iterate through sibling_weights
for (k0, v0) in outcomes.items():
    # iterate through parent_weights
    for (k1, v1) in v0.items():
        # iterate through alphas
        for (k2, v2) in v1.items():
            #iterate through betas
            for (k3, v3) in v2.items():
                text_values += '%1.10f, %2.10f, %1.1f, %1.1f, %1.1f, %1.1f\n' \
                                %(v3['intensity'], v3['gradient'], k0, k1, k2, k3)
print text_values
f = open('outcomes.csv', 'w')
f.write(text_values)
f.close()