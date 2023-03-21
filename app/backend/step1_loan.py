from pydantic import BaseModel, validator

EMP_LENGTH_MAPPING = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8, 
    '9 years': 9,
    '10+ years': 10
}

PURPOSE_VALUES = {
    'debt_consolidation': 'debt consolid',
    'small_business': 'small busi',
    'home_improvement': 'home improv',
    'major_purchase': 'major purchas',
    'credit_card': 'credit card',
    'other': 'other', 
    'house': 'hous', 
    'vacation': 'vacat',
    'car': 'car', 
    'medical': 'medic', 
    'moving': 'move',
    'renewable_energy': 'renew energi',
    'wedding': 'wed',
    'educational': 'educ'
}

class LoanStep1(BaseModel):
    '''
        loan_amnt float64
        dti float64
        emp_length float64
        purpose object
    '''

    loan_amnt: float = 2500.0
    dti: float = 0.308
    emp_length: str = "5 years"
    purpose: str = "other"

    @validator("loan_amnt")
    def loan_amnt_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError(f"expected loan_amnt >= 0, received value - {value}")
        return value

    @validator("dti")
    def dti_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError(f"expected dti >= 0, received value - {value}")
        return value

    @validator("emp_length")
    def emp_length_must_have_value(cls, value):
        if value not in EMP_LENGTH_MAPPING.keys():
            raise ValueError(f"expected emp_length values are {list(EMP_LENGTH_MAPPING.keys())}. Received value - {value}")
        return value
   
    @validator("purpose")
    def purpose_must_have_value(cls, value):
        if value not in PURPOSE_VALUES.keys():
            raise ValueError(f"expected purpose values are {list(PURPOSE_VALUES.keys())}. Received value - {value}")
        return value
   
    def get_entry_dict(self):
        data = {}

        data["loan_amnt"] = [self.loan_amnt]
        data["dti"] = [self.dti]
        data["emp_length"] = [EMP_LENGTH_MAPPING[self.emp_length]]
        data["purpose"] = [PURPOSE_VALUES[self.purpose]]
 
        return data