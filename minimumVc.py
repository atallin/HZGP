import numpy as np

options = [
     {'label':'Constant Vc', 'value':'constantvc'},
     {'label':'Oroskar&Turian', 'value':'oroskar'}
    ]
default_option = options[0]['value']

def mintransportvelocity(model, **kwarg):
    for k in kwarg.keys():
        if k == 'C':
            C= kwarg[k]         # vol/vol  volume concentration of proppant
        elif k == 'rho':
            rho = kwarg[k]      # lb/gal
        elif k == 'DA':
            DA = kwarg[k]       # in
        elif k == 'bP':
            bP = kwarg[k]       # in/in bed length/wetted perimeter
        elif k == 'SG':
            SG = kwarg[k]       # g/ml proppant specific gravity
        elif k == 'dp':
            dp = kwarg[k]       # in proppant diameter
        elif k == 'mu':
            mu = kwarg[k]       # cP fluid viscosity
    
    if model == 'oroskar':
        modelparms = {'A': 1.85,
                      'C': 0.1536,
                      'C1': 0.3564,
                      'd/DA': -0.378,
                      'NRe': 0.09
                      }
        if not all([k in kwarg.keys() for k in ['C','rho','DA', 'SG', 'dp', 'mu']]):
            return None
        S = 8.3454045 * SG/rho                         # --
        dgS1 = 0.28867513 * np.sqrt(dp*32.174049*(S-1))
        NRe = 927.68661 * rho * DA * dgS1 /mu          # -- unit conversion lb/gal * in * ft/sec / cP = 927.68661
        vtemp = modelparms['A'] * \
                ((1-C)**modelparms['C1']) * \
                ((C)**modelparms['C']) * \
                ((dp/DA)**modelparms['d/DA'])* \
                (NRe ** modelparms['NRe'])
        return vtemp * dgS1
    elif model == 'constantvc':
        if not 'vc' in kwarg.keys():
            return None
        return kwarg['vc']
    return None        
