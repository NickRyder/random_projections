import scipy.optimize as spo
import numpy as np

#Package used for testing orp methods



# General use instructions:
# Call objective_minimize with diagonal entries and desired lambda

#TODO: add documentation and input checking


#Takes as args (d,lamb,q)
#d - sorted list of singular values
#lamb - positive scalar
#q - exponent of expected bias^2
def objective(a, args):
    a = np.array(a)
    d = np.array(args[0])
    lamb = args[1]
    # q = args[2]
    return np.sum((a**2-d)**2) + lamb*(np.sum(a**4)+(np.sum(a**2)**2))


def objective_minimizer(d, lamb):
    n = len(d)
    return spo.minimize(objective, np.ones(n), args=[d, lamb])



#Numerical optimizer to determine mix of random and deterministic
def objective_mixed(a, args):
    d, lamb, est_N, target_dim = args[0], args[1], args[2], args[3]
    rand_est_N = est_N
    determ_est_N = target_dim - rand_est_N

    if determ_est_N == 0:
        arg = np.array(a)
        return np.sum((arg**2 - d)**2) + (lamb/rand_est_N)*(np.sum(arg**4)+(np.sum(arg**2)**2))
    elif rand_est_N == 0:
        arg = np.array(a[:-determ_est_N])
        determ = np.pad(np.array(a[-determ_est_N:]), (len(arg)-determ_est_N, 0), 'constant')
        return np.sum((d + determ)**2) + (np.sum(arg**4)+(np.sum(arg**2)**2))
    else:
        arg = np.array(a[:-determ_est_N])
        determ = np.pad(np.array(a[-determ_est_N:]), (len(arg)-determ_est_N, 0), 'constant')
        return np.sum((arg**2 - d + determ)**2)+ (lamb/rand_est_N)*(np.sum(arg**4)+(np.sum(arg**2)**2))


#WARNING: only works for sorted d
def objective_minimizer_mixed(d, lamb, target_dim):
    n = len(d)
    obj_min = float('inf')
    scipy_minimizer_min = None
    for est_N in range(0, target_dim+1):
        minimizer = spo.minimize(objective_mixed, np.ones(n+target_dim-est_N), args=[d, lamb, est_N, target_dim])
        if minimizer['fun'] < obj_min:
            obj_min = minimizer['fun']
            if target_dim - est_N > 0:
                minimizer['x'] = minimizer['x'][:-target_dim + est_N]

            scipy_minimizer_min = minimizer
    return scipy_minimizer_min






#Solve for nu
# def nuSolver(d, lamb, q, m):
#     if q is 1:
#         return 1
    # elif q == .5:
    #     return spo.newton(halfObjective,1,args=[d,lamb,m])
    #
    # else:
    #     return spo.fixed_point(nuObjective, 0, args=[d, lamb, q, m])




#args = [d,lamb,q,m]
# def nuObjective(nu, *args):
#     d = args[0]
#     lamb = args[1]
#     q = args[2]
#     m = args[3]
#
#
#     if m > 0:
#         firstSummand = np.sum((d*d)[:-m])
#         dSigma = np.sum(d[-m:])
#         secondSummandList = lamb*(d[-m:]*(lamb + lamb*m+nu)+nu*dSigma)/((lamb+nu)*(lamb+lamb*m+nu))
#     else:
#         firstSummand = np.sum((d*d))
#         secondSummandList = 0
#
#     secondSummand = np.sum(secondSummandList ** 2)
#     return q*((firstSummand + secondSummand)** (q-1))


# def halfObjective(nu, *args):
#     d = args[0]
#     lamb = args[1]
#     m = args[2]
#
#     denom = ((lamb + nu)*(lamb + lamb * m + nu))**2
#
#     if m > 0:
#         firstSummand = np.sum((d*d)[:-m])
#         dSigma = np.sum(d[-m:])
#         secondSummandList = (lamb*d[-m:]*(lamb + lamb*m+nu)+lamb*nu*dSigma)
#     else:
#         firstSummand = np.sum((d*d))
#         secondSummandList = 0
#
#     secondSummand = np.sum(secondSummandList ** 2)
#     return 4*nu*nu*(firstSummand*denom+secondSummand) - denom



