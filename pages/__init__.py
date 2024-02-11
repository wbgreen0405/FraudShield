# Inside pages/__init__.py

# Import the app function from each page module
from .home import app as home_app
from .transactions import app as transactions_app
from .approval_system import app as approval_system_app
from .anomaly_detection import app as anomaly_detection_app
from .case_detail import app as case_detail_app
from .test_and_learn_loop_page import app as test_and_learn_loop_app
from .help_documentation_page import app as help_documentation_app
from .audit_logs_history_page import app as audit_logs_history_app

# Optionally, create a dictionary mapping page names to their app functions
# This can be useful if you want to programmatically access the app functions
pages = {
    "Home": home_app,
    "Transaction Analysis": transactions_app,
    "Approval System": approval_system_app,
    "Anomaly Detection": anomaly_detection_app,
    "Case Detail": case_detail_app,
    "Test and Learn Loop": test_and_learn_loop_app,
    "Help / Documentation": help_documentation_app,
    "Audit Logs / History": audit_logs_history_app,
    
}

