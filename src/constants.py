VARIABLE_TYPE = {
    "ATTEND": "categorical",
    "BFACIL": "categorical",
    "BMI": "continuous",
    "CIG_0": "integer",
    "DLMP_MM": "integer (cyclic)",
    "DMAR": "categorical",
    "DOB_MM": "integer (cyclic)",
    "DOB_TT": "timestamp (cyclic)",
    "DOB_WK": "integer (cyclic)",
    "FAGECOMB": "integer",
    "FEDUC": "categorical (ordinal)",
    "ILLB_R": "integer",
    "ILOP_R": "integer",
    "ILP_R": "integer",
    "LD_INDL": "categorical",
    "MAGER": "integer",
    "MBSTATE_REC": "categorical",
    "MEDUC": "categorical (ordinal)",
    "M_Ht_In": "integer",
    "NO_INFEC": "categorical",
    "NO_MMORB": "categorical",
    "NO_RISKS": "categorical",
    "PAY": "categorical",
    "PAY_REC": "categorical",
    "PRECARE": "integer",
    "PREVIS": "integer",
    "PRIORDEAD": "integer",
    "PRIORLIVE": "integer",
    "PRIORTERM": "integer",
    "PWgt_R": "continuous",
    "RDMETH_REC": "categorical",
    "RESTATUS": "categorical",
    "RF_CESAR": "categorical",
    "RF_CESARN": "integer",
    "SEX": "categorical",
    "WTGAIN": "categorical",
}

MISSING_CODE = {
    "ATTEND": "9",
    "BFACIL": "9",
    "BMI": 99.9,
    "CIG_0": 99,
    "DLMP_MM": 99,
    "DMAR": " ",
    "DOB_MM": None,
    "DOB_TT": 9999,
    "DOB_WK": None,
    "FAGECOMB": 99,
    "FEDUC": "9",
    "ILLB_R": 999,
    "ILOP_R": [888, 999],
    "ILP_R": 999,
    "LD_INDL": None,
    "MAGER": None,
    "MBSTATE_REC": "3",
    "MEDUC": "9",
    "M_Ht_In": 99,
    "NO_INFEC": "9",
    "NO_MMORB": "9",
    "NO_RISKS": "9",
    "PAY": "9",
    "PAY_REC": "9",
    "PRECARE": 99,
    "PREVIS": 99,
    "PRIORDEAD": 99,
    "PRIORLIVE": 99,
    "PRIORTERM": 99,
    "PWgt_R": 999,
    "RDMETH_REC": "9",
    "RESTATUS": None,
    "RF_CESAR": "U",
    "RF_CESARN": 99,
    "SEX": None,
    "WTGAIN": "99",
}
