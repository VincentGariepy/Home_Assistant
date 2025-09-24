
import os

# Dictionary of predefined addresses accessed through environment variables
ADDRESSES = {
    'home_address':os.getenv('GOOGLE_API_HOME_ADDRESS'),
    'olivia_address':os.getenv('GOOGLE_API_OLIVIA_ADDRESS'),
    'chantal_address':os.getenv('GOOGLE_API_CHANTAL_ADDRESS'),
    'gilles_address':os.getenv('GOOGLE_API_CHANTAL_ADDRESS'),
    'roxanne_address':os.getenv('GOOGLE_API_ROXANNE_ADDRESS'),
    'josee_address':os.getenv('GOOGLE_API_JOSEE_ADDRESS'),
    'andre_address':os.getenv('GOOGLE_API_ANDRE_ADDRESS'),
}