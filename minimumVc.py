import numpy as np

options = [
    {'label':'Model 1', 'value':'model1'},
    {'label':'Model 2', 'value':'model2'},
    {'label':'Conventional',  'value':'conventional'},
    {'label':'linear', 'value':'linear'},
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
    
    if model == 'model1':
        modelparms = {'A':      6.14,
                      'C':      0.415,
                      'bP':     -0.261,
                      'd/DA':   -0.360,
                      'NRe':    0.0381
                      }
        if not all([k in kwarg.keys() for k in ['C','rho','DA', 'bP', 'SG', 'dp', 'mu']]):
            return None
        S = 8.3454045 * SG/rho                         # --
        dgS1 = 0.28867513 * np.sqrt(dp*32.174049*(S-1)) # ft/sec
        NRe = 927.68661 * rho * DA * dgS1 /mu          # -- unit conversion lb/gal * in * ft/sec / cP = 927.68661
        vtemp = modelparms['A'] * \
                (C**modelparms['C']) * \
                (bP**modelparms['bP']) * \
                ((dp/DA)**modelparms['d/DA'])* \
                (NRe ** modelparms['NRe'])
        return vtemp*dgS1
    elif model == 'model2':
        modelparms = {'A':      1.35,
                      'C':      -3.77,
                      'bP':     -0.243,
                      'd/DA':   -0.37,
                      'NRe':    0.0426
                      }
        if not all([k in kwarg.keys() for k in ['C','rho','DA', 'bP', 'SG', 'dp', 'mu']]):
            return None
        S = 8.3454045 * SG/rho                         # --
        dgS1 = 0.28867513 * np.sqrt(dp*32.174049*(S-1))
        NRe = 927.68661 * rho * DA * dgS1 /mu          # -- unit conversion lb/gal * in * ft/sec / cP = 927.68661
        # print(f'NRe: {NRe}, rho: {rho}, DA: {DA}, dgS1: {dgS1} mu: {mu}, dp: {dp}, SG: {SG}, C: {C}, bP: {bP}')
        vtemp = modelparms['A'] * \
                ((1-C)**modelparms['C']) * \
                (bP**modelparms['bP']) * \
                ((dp/DA)**modelparms['d/DA'])* \
                (NRe ** modelparms['NRe'])
        return vtemp * dgS1
    elif model == 'conventional': 
        # 1.15915872803019,//C1 
        # -3.58822426562324, /*n_1 */ -0.377485123473551,  //n_2
        # 0.05439676041480,  /*n_3 */ -0.272080143824705 /* n_4 */
        modelparms = {'A': 1.15915872803019,
                      'C': -3.58822426562324,
                      'd/DA': -0.377485123473551,
                      'NRe': 0.05439676041480,
                      'bP': -0.272080143824705
                      } 
        if not all([k in kwarg.keys() for k in ['C','rho','DA', 'bP', 'SG', 'dp', 'mu']]):
            return None
        S = 8.3454045 * SG/rho                         # -- 
        dgS1 = 0.28867513 * np.sqrt(dp*32.174049*(S-1))
        NRe = 927.68661 * rho * DA * dgS1 /mu
        vtemp = modelparms['A'] * \
                ((1-C)**modelparms['C']) * \
                (bP**modelparms['bP']) * \
                ((dp/DA)**modelparms['d/DA']) * \
                (NRe ** modelparms['NRe'])
        return vtemp * dgS1
    elif model == 'oroskar':
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
    elif model == 'linear':
        modelparms = {'A': 4.363474936,
                      'C': 47.06887855,
                      'mu': -0.068298946,
                      'dp': 60.01919729,
                      'rho':-0.992211556,
                      'DA':1.301178557
                      }
        
        if not all([k in kwarg.keys() for k in ['C','rho','DA', 'SG', 'dp', 'mu']]):
            return None
        Vc = modelparms['A']  \
            + (C*modelparms['C']) \
            + (mu*modelparms['mu']) \
            + (dp*modelparms['dp']) \
            + (rho*modelparms['rho']) \
            + (DA*modelparms['DA'])
        return Vc
    elif model == 'constantvc':
        if not 'vc' in kwarg.keys():
            return None
        return kwarg['vc']
    return None        

