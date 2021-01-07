import scipy
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)
import csv
from scipy.optimize import minimize
from math import *
from scipy import stats


""""Function impData imports the raw data from a csv file and returns all data in list of string elements"""
def impData(filename):
    abs_file_path = filename
    with open(abs_file_path, newline='') as csvfile:
        totalData = list(csv.reader(csvfile))
    return totalData

def expVariance(vals):
    diff = []
    for l in range(len(vals)):
        diff.append((vals[l]-np.average(vals))**2)
    var = sum(diff)/(len(diff)-1)
    print(vals, var)
    return var

def filterData(data):
    concentrations = []
    responses = []
    weights = []
    proportions = []
    numDrugs = int((len(data[0])-2)/3)

    for j in range(numDrugs+1):

        concentration = list(np.array(data)[1:, j * 2])  # concentration (x-axis) of drug
        concentration = list(filter(None, concentration))
        concentration = [float(y) for y in concentration]
        response = list(np.array(data)[1:, j * 2 + 1])
        response = list(filter(None, response))
        response = [float(y) for y in response]



        tempConc = [concentration[0]]
        meanResp = []
        weightResp = []
        initiate = 0
        indexList = [0]
        prop = []

        for k in range(1,len(concentration)):

            if concentration[k] != concentration[k-1]:
                indexList.append(k)
                tempConc.append(concentration[k])
                meanResp.append(np.average(response[initiate:k]))
                weightResp.append(1/expVariance(response[initiate:k]))
                initiate = k
        meanResp.append(np.average(response[initiate:]))
        weightResp.append(1 / expVariance(response[initiate:]))


        if j > 0:
            prop = list(np.array(data)[1:, -(numDrugs + 1 - j)])
            prop = list(filter(None, prop))
            prop = [float(y) for y in prop]


        concentrations.append(tempConc)
        responses.append(meanResp)
        weights.append(weightResp)
        proportions.append(prop)



    finalProps = []
    proportions = proportions[1:]
    transProp = np.transpose(proportions).tolist()
    for x in indexList:
        finalProps.append(transProp[x])

    fixedRatio = True
    for f in range(len(proportions)):
        if len(set(proportions[f])) != 1:
            fixedRatio = False

    standardWeights = []
    for i in range(len(concentrations)):
        indWeights = []
        for j in range(len(concentrations[i])):
            spot = concentrations[i][j]
            tempReps = []
            for k in range(len(concentrations[i])):
                if concentrations[i][k] == spot:
                    tempReps.append(weights[i][k])
            stdWeight = weights[i][j]/sum(tempReps)
            indWeights.append(stdWeight)
        standardWeights.append(indWeights)

    finalConcentrations = []
    finalResponses = []
    finalWeights = []


    for i in range(numDrugs):
        totalData = [concentrations[i], responses[i], weights[i]]
        totalData = np.transpose(totalData)
        totalData = sorted(totalData, key=lambda l: float(l[0]))
        totalData = np.transpose(totalData).tolist()
        finalConcentrations.append(totalData[0])
        finalResponses.append(totalData[1])
        finalWeights.append(totalData[2])

    totalData = [concentrations[-1], responses[-1], weights[-1]] + [finalProps]
    totalData = np.transpose(totalData)
    totalData = sorted(totalData, key=lambda l: float(l[0]))
    totalData = np.transpose(totalData).tolist()

    finalConcentrations.append(totalData[0])
    finalResponses.append(totalData[1])
    finalWeights.append(totalData[2])
    finalProps = totalData[3]
    finalProps = list(np.array(finalProps))



    return finalConcentrations, finalResponses, finalWeights, finalProps, fixedRatio



"""Function hill imports a starting value for 3 parameters (thetaMax, hill coefficient, and EC50) and
returns the sum of the residuals squared to be minimized"""
def hill_4(x, t, y, w):
     return sum(w[i]*(x[3]+x[0]*t[i]**x[1]/(x[2]**x[1]+t[i]**x[1]) - y[i])**2 for i in range(len(t)))


def hill_3(x, t, y, w):
    return sum(w[i] * (x[0] * t[i] ** x[1] / (x[2] ** x[1] + t[i] ** x[1]) - y[i]) ** 2 for i in range(len(t)))

"""Function plotIndFits imports parameters, concentration and response values, and a binary value for whether
to plot on a semilog plot and plots the fitted curve with the data"""
def plotIndFits_4(params, conc, resp, logging, j):
    index = np.linspace(min(conc), max(conc), 50000)
    func = params[3]+ params[0]*index**params[1]/(params[2]**params[1]+index**params[1])
    plt.plot(index, func)
    plt.scatter(conc,resp)
    if logging == 1:
        plt.xscale('log')

    plt.xlabel('Drug Concentration (uM)')
    plt.ylabel('Fractional Response')
    plt.title('Empirical Fit: Drug ' +str(j+1))

    plt.show()

def plotIndFits_3(params, conc, resp, logging, j):
    index = np.linspace(min(conc), max(conc), 50000)
    func = params[0]*index**params[1]/(params[2]**params[1]+index**params[1])
    plt.plot(index, func)
    plt.scatter(conc,resp)
    if logging == 1:
        plt.xscale('log')

    plt.xlabel('Drug Concentration (uM)')
    plt.ylabel('Fractional Response')
    plt.title('Empirical Fit: Drug ' +str(j+1))

    plt.show()

def plotComparisons(x, theoretical, theoError, observed, obsError,logging, csv):
    xList = []
    for i in range(len(x)):
        xList.append(x[i][0])

    x = xList

    plt.errorbar(x, theoretical, yerr=theoError, fmt='o', color='black',
                 ecolor='lightgray', elinewidth=3, capsize=0, label='Expected')
    plt.errorbar(x, observed, yerr=obsError, fmt='o', color='blue',
                ecolor='lightskyblue', elinewidth=3, capsize=0,label='Empirical')
    plt.ylim(0,1)
    """if logging == 1:
        plt.xscale('log')"""
    plt.xlabel('Drug Concentration (uM)')
    plt.ylabel('Fractional Response')
    plt.legend()
    plt.title(csv)
    plt.show()


def plotComboFits(fits, minC, maxC, logging, combo):
    indFuncs = []
    index = np.linspace(minC, maxC, 50000)
    for f in range(len(fits)):
        params = fits[f]
        indFuncs.append(params[0] * index ** params[1] / (params[2] ** params[1] + index ** params[1]))

    for a in range(len(indFuncs)):
        plt.plot(index, indFuncs[a], label = 'Drug ' + str(a+1))
    conc = index
    plt.plot(index, eval(combo),label = 'Combo', linestyle='--')
    if logging == 1:
        plt.xscale('log')
    plt.legend()
    plt.xlabel('Drug Concentration (nM)')
    plt.ylabel('Fractional Response')
    plt.title('Individual Fits and Theoretical Combined Curve')

    plt.show()

def plotComboError(comboError, minC, maxC, logging):

    index = np.linspace(minC, maxC, 50000)
    results = []
    for i in range(1,len(index)):
        conc = index[i]
        results.append(eval(comboError))

    plt.plot(index[1:], results,label = 'Combo Error')
    if logging == 1:
        plt.xscale('log')
    plt.legend()

    plt.show()

def jackknife(fits, pseudoFits):
    residA = []
    residB = []
    residC = []
    sumA = 0
    sumB = 0
    sumC = 0
    tot = len(pseudoFits)
    denom = tot * (tot - 1)
    averageA = averageB = averageC = 0

    for b in range(tot):
        averageA += pseudoFits[b][0]/tot
        averageB += pseudoFits[b][1]/tot
        averageC += pseudoFits[b][2]/tot

    for s in range(len(pseudoFits)):
        residA.append(tot * averageA - (tot - 1) * pseudoFits[s][0])
        residB.append(tot * averageB - (tot - 1) * pseudoFits[s][1])
        residC.append(tot * averageC - (tot - 1) * pseudoFits[s][2])

    #print(residA)
    for h in range(len(residA)):
        sumA += (residA[h] - np.average(residA)) ** 2
        sumB += (residB[h] - np.average(residB)) ** 2
        sumC += (residC[h] - np.average(residC)) ** 2

    errors = [np.sqrt(sumA / denom), np.sqrt(sumB / denom), np.sqrt(sumC / denom)]

    return errors

def comboEquation(fits, drugRatio):
    equations = []
    for f in range(len(fits)):
        p = fits[f]
        equations.append(str(p[0]) + '*' + str(drugRatio[f]) + '*conc**' + str(p[1]) + '/(' + str(p[2])+ '**' + str(p[1]) + '+' + str(drugRatio[f]) + '*conc**' + str(p[1]) + ')')
    bigEquation = str(equations[0])

    for e in range(1, len(equations)):
        bigEquation += '-' + equations[e] + '*(' + bigEquation + '-1)'


    return(bigEquation)

def excludeOneEquations(fits, drugRatio):
    equations = []
    excludedEquations = []
    index = []
    for f in range(len(fits)):
        p = fits[f]
        index.append(f)
        equations.append(str(p[0]) + '*(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + '/(' + str(p[2]) + '**' + str(
            p[1]) + '+(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + ')')

    for e in range(len(equations)):
        iteration = index[0:e] + index[e+1:]
        excludeEquation = str(equations[iteration[0]])
        for g in iteration[1:]:
            excludeEquation += '-' + equations[g] + '*(' + excludeEquation + '-1)'
        excludedEquations.append(excludeEquation)

    return(excludedEquations)


def comboError(fits, drugRatio, paramError):
    partialDerivC = []
    partialDerivN = []
    partialDerivS = []

    excluded = excludeOneEquations(fits, drugRatio)

    for f in range(len(fits)):
        conc=400
        p = fits[f]
        partialDerivC.append('(-' + str(p[1]) + '*' + str(p[0]) + '*' + str(p[2]) + '**(' + str(p[1])+ '-1)' + '*(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + '/(' + str(p[2]) + '**' + str(
            p[1]) + '+(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + ')**2*' + '(1-' + excluded[f] + '))**2*' + str(paramError[f][2]) + '**2')
        partialDerivN.append('(' + str(p[0]) + '*' + str(p[2]) + '**' + str(p[1]) + '*(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + '*(log(conc*' + str(drugRatio[f]) + ')-log(' + str(p[2]) + '))/(' + str(p[2]) + '**' + str(
            p[1]) + '+(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + ')**2*'  + '(1-' + excluded[f] + '))**2*'+ str(paramError[f][1]) + '**2')
        partialDerivS.append('((' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + '/(' + str(p[2]) + '**' + str(
            p[1]) + '+(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + ')*'  + '(1-' + excluded[f] + '))**2*' + str(paramError[f][0]) + '**2')

    allC = partialDerivC[0]
    allN = partialDerivN[0]
    allS = partialDerivS[0]


    for t in range(1, len(partialDerivC)):
        allC += '+' + partialDerivC[t]
        allN += '+' + partialDerivN[t]
        allS += '+' + partialDerivS[t]


    totalError = 'sqrt(' + allC + '+' + allN + '+' + allS + ')'



    return(totalError)

def plotComboAll(concen, resp, combinedEffect, upperConfidence, lowerConfidence, logging):
    index = np.linspace(concen[0], concen[-1], 50000)


    upper = []
    lower = []
    for i in range(1, len(index)):
        conc = index[i]
        upper.append(eval(upperConfidence))
        lower.append(eval(lowerConfidence))
    conc=index

    plt.plot(index, eval(combinedEffect), label='Theoretical Independence')
    plt.plot(index[1:], upper, color='k', linestyle='--')
    plt.plot(index[1:], lower, color='k', linestyle='--')
    plt.scatter(concen, resp)
    if logging == 1:
        plt.xscale('log')

    plt.xlabel('Drug Concentration (nM)')
    plt.ylabel('Fractional Response')
    plt.title('Drug Combo Empirical vs Theoretical Fit')

    plt.show()

def allDeviation(fits, drugRatio, paramError, numPoints):
    partialDerivC = []
    partialDerivN = []
    partialDerivS = []

    excluded = excludeOneEquations(fits, drugRatio)

    for f in range(len(fits)):
        p = fits[f]
        partialDerivC.append(
            '(-' + str(p[1]) + '*' + str(p[0]) + '*' + str(p[2]) + '**(' + str(p[1]) + '-1)' + '*(' + str(
                drugRatio[f]) + '*conc)**' + str(p[1]) + '/(' + str(p[2]) + '**' + str(
                p[1]) + '+(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + ')**2*' + '(1-' + excluded[
                f] + '))**2*' + str(paramError[f][2]) + '**2*' + str(numPoints[f]))
        partialDerivN.append(
            '(' + str(p[0]) + '*' + str(p[2]) + '**' + str(p[1]) + '*(' + str(drugRatio[f]) + '*conc)**' + str(
                p[1]) + '*(log(conc*' + str(drugRatio[f]) + ')-log(' + str(p[2]) + '))/(' + str(p[2]) + '**' + str(
                p[1]) + '+(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + ')**2*' + '(1-' + excluded[
                f] + '))**2*' + str(paramError[f][1]) + '**2*' + str(numPoints[f]))
        partialDerivS.append('((' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + '/(' + str(p[2]) + '**' + str(
            p[1]) + '+(' + str(drugRatio[f]) + '*conc)**' + str(p[1]) + ')*' + '(1-' + excluded[f] + '))**2*' + str(
            paramError[f][0]) + '**2*' + str(numPoints[f]))
    allC = partialDerivC[0]
    allN = partialDerivN[0]
    allS = partialDerivS[0]

    for t in range(1, len(partialDerivC)):
        allC += '+' + partialDerivC[t]
        allN += '+' + partialDerivN[t]
        allS += '+' + partialDerivS[t]

    totalDeviation = 'sqrt(' + allC + '+' + allN + '+' + allS + ')'

    return (totalDeviation)




def main():

    #user input whether semilog or not (values = 0 or 1)
    logging = 1
    interceptCorrection = 1
    csvname = "HCC1937_OlapB02"
    file = "C:/Users/Richard/Desktop/DrugCombosKalin/updatedFiles_Nov2020/" + csvname + ".csv"  # import data file
    concentrations, responses, weights, proportions, fixedRatio = filterData(impData(file))
    minConcentration = 1000000
    maxConcentration = 0
    fits = []
    indError = []
    numPoints = []

    for j in range(len(concentrations)-1):  # iterate through the number of data sets (number of data columns / 3)
        concentration = concentrations[j]  # concentration (x-axis) of drug
        response = responses[j]  # response (y-axis) to drug
        weight = weights[j]
        numPoints.append(len(response))
        if float(min(concentration)) < minConcentration:
            minConcentration = float(min(concentration))
        if float(max(concentration)) > maxConcentration:
            maxConcentration = float(max(concentration))


        x0 = np.array([0.9, 1.0, 0.05, 0])  # initial guess for curve fit parameters
        x0_red = np.array([0.9, 1.0, 0.05])
        tol = 1e-40
        # weighted least squares nonlinear regression with bounds
        res = minimize(hill_4, x0, args=(concentration, response, weight), method='SLSQP',
                       bounds=((0, 1), (None, None), (0, None), (None, None)), tol=None,
                       options={'maxiter': 10000, 'ftol': tol})

        params = res.x  # give parameters a variable name
        if interceptCorrection == 0:
            params_reduced = [params[0], params[1], params[2]]
        else:
            for c in range(len(response)):
                response[c] = (params[3] - response[c]) / (params[3] - 1)
            res = minimize(hill_3, x0_red, args=(concentration, response, weight), method='BFGS', tol=None,
                           options={'maxiter': 10000, 'ftol': tol})
            params = res.x
            rss = hill_3(params, concentration, response, weight) / (len(concentration) - len(params))
            #print(res.hess_inv)
            covariance = np.sqrt(np.diag(res.hess_inv)) * np.sqrt(rss)

            #print(covariance)
            params_reduced = [params[0], params[1], params[2]]

        fits.append(params_reduced)  # add curve parameters to fits list
        #print(params)
        if interceptCorrection == 0:
            plotIndFits_4(params, concentration, response, logging, j)  # plot curves with data
        else:
            plotIndFits_3(params, concentration, response, logging, j)  # plot curves with data

        pseudoFits = []
        for r in range(len(concentration)):
            pseudoWeight = weight[0:r] + weight[r + 1:]
            pseudoConcentration = concentration[0:r] + concentration[r + 1:]
            pseudoResponse = response[0:r] + response[r + 1:]

            if interceptCorrection == 0:
                pseudoRes = minimize(hill_4, x0, args=(pseudoConcentration, pseudoResponse, pseudoWeight),
                                     method='SLSQP',
                                     bounds=((0, 1), (None, None), (0, None), (None, None)), tol=None,
                                     options={'maxiter': 100000, 'ftol': 1e-40})
            else:
                pseudoRes = minimize(hill_3, x0_red, args=(pseudoConcentration, pseudoResponse, pseudoWeight),
                                     method='SLSQP',
                                     bounds=((0, 1), (None, None), (0, None)), tol=None,
                                     options={'maxiter': 100000, 'ftol': 1e-40})

            pseudoparams = pseudoRes.x  # give parameters a variable name

            if interceptCorrection == 0:
                pseudoparams_reduced = [pseudoparams[0], pseudoparams[1], pseudoparams[2], pseudoparams[3]]
            else:
                pseudoparams_reduced = [pseudoparams[0], pseudoparams[1], pseudoparams[2]]

            pseudoFits.append(pseudoparams_reduced)  # add curve parameters to fits list

            # plotIndFits(pseudoFits[r], pseudoConcentration, pseudoResponse, logging, r)

        # print(pseudoFits)
        paramError = jackknife(params_reduced, pseudoFits)
        indError.append(paramError)
        #print(paramError)
    drugRatioList = proportions
    concentrationCombo = concentrations[-1]  # concentration (x-axis) of drug
    responseCombo = responses[-1]  # response (y-axis) to drug
    weightCombo = weights[-1]
    theoreticalResponseCombo = []
    theoreticalError = []
    theoreticalGobalN = []
    weightSumCombo = 0


    if float(min(concentrationCombo)) < minConcentration:
        minConcentration = float(min(concentrationCombo))
    if float(max(concentrationCombo)) > maxConcentration:
        maxConcentration = float(max(concentrationCombo))


    """if fixedRatio == True:
        combinedEffect = comboEquation(fits, proportions[0])
        allError = comboError(fits, proportions[0], indError)
        upperConfidence = combinedEffect + '+1.96*' + allError
        lowerConfidence = combinedEffect + '-1.96*' + allError
        plotComboAll(concentrationCombo, responseCombo, combinedEffect, upperConfidence, lowerConfidence, logging)"""

    groupedConcentrationCombo = [[concentrationCombo[0]]]
    groupedWeightCombo = [[weightCombo[0]]]
    groupedResponseCombo = [[responseCombo[0]]]
    groupedDrugRatio = [drugRatioList[0]]


    for l in range(1, len(concentrationCombo)):
        if (concentrationCombo[l] == concentrationCombo[l - 1]) and (drugRatioList[l].all() == drugRatioList[l - 1].all()):
            groupedConcentrationCombo[-1].append(concentrationCombo[l])
            groupedWeightCombo[-1].append(weightCombo[l])
            groupedResponseCombo[-1].append(responseCombo[l])


        else:
            groupedConcentrationCombo.append([concentrationCombo[l]])
            groupedWeightCombo.append([weightCombo[l]])
            groupedResponseCombo.append([responseCombo[l]])
            groupedDrugRatio.append(drugRatioList[l])


    groupedSE = []
    groupedAvg = []
    for u in range(len(groupedWeightCombo)):
        print(len(groupedWeightCombo))
        drugRatio = groupedDrugRatio[u]

        combinedEffect = comboEquation(fits, drugRatio)
        conc = float(groupedConcentrationCombo[u][0])
        theoreticalResponseCombo.append(eval(combinedEffect))

        # print(combinedEffect)

        # plotComboFits(fits, minConcentration, maxConcentration, logging, combinedEffect)
        allError = comboError(fits, drugRatio, indError)
        theoreticalError.append(eval(allError))
        # print(allError)
        # plotComboError(allError, minConcentration, maxConcentration, logging)

        globalN = '(' + allDeviation(fits, drugRatio, indError, numPoints) + '/' + allError + ')**2'
        theoreticalGobalN.append(eval(globalN))

        sumEach = sum(groupedWeightCombo[u])
        dev = []
        avgResponse = np.average(groupedResponseCombo[u])
        groupedAvg.append(avgResponse)
        for t in range(len(groupedWeightCombo[u])):
            groupedWeightCombo[u][t] = groupedWeightCombo[u][t] / sumEach
            dev.append(groupedWeightCombo[u][t] ** 2 * (groupedResponseCombo[u][t] - avgResponse) ** 2)
        SE = np.sqrt(len(groupedWeightCombo[u]) / (len(groupedWeightCombo[u]) - 1) * sum(dev))
        groupedSE.append(SE)

    """print(groupedConcentrationCombo)
    print(groupedWeightCombo)
    print(groupedResponseCombo)
    print(groupedSE)"""

    t_stat = []
    df = []
    pWeights = []
    for i in range(len(groupedSE)):
        conc = groupedConcentrationCombo[i][0]
        if conc <= 0:
            print('Group ' + str(i + 1) + ' concentration is not greater than 0, so it cannot be evaluated.')

        else:

            t_val = (groupedAvg[i] - theoreticalResponseCombo[i]) / np.sqrt(
                groupedSE[i] ** 2 / len(groupedWeightCombo[i]) + theoreticalError[i] ** 2 / theoreticalGobalN[i])
            t_stat.append(t_val)

            v = (groupedSE[i] ** 2 / len(groupedWeightCombo[i]) + theoreticalError[i] ** 2 / theoreticalGobalN[
                i]) ** 2 / (groupedSE[i] ** 4 / (len(groupedWeightCombo[i]) ** 2 * (len(groupedWeightCombo[i]) - 1)) +
                            theoreticalError[i] ** 4 / (theoreticalGobalN[i] ** 2 * (theoreticalGobalN[i] - 1)))
            df.append(v)

            pWeights.append(np.sqrt(len(groupedWeightCombo[i])))

    pValsSynergy = []
    pValsAntagonism = []
    for d in range(len(df)):
        pValsSynergy.append(stats.t.sf(t_stat[d], df[d]))
        pValsAntagonism.append(stats.t.sf(-t_stat[d], df[d]))

    print(pValsSynergy)
    print(pValsAntagonism)

    totalPSyn = scipy.stats.combine_pvalues(pValsSynergy, method='stouffer', weights=pWeights)
    totalPAnt = scipy.stats.combine_pvalues(pValsAntagonism, method='stouffer', weights=pWeights)

    print(totalPSyn)
    print(totalPAnt)
    # print(groupedConcentrationCombo)
    plotComparisons(groupedConcentrationCombo, theoreticalResponseCombo, theoreticalError, groupedAvg, groupedSE,
                    logging, csvname)


main()