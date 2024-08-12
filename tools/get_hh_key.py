import requests

def generate_oauth_url(client_id, redirect_uri):
    """
    Generates the OAuth URL for authorization.

    Args:
    client_id (str): The client ID of your application.
    redirect_uri (str): The redirect URI for your application.

    Returns:
    str: The OAuth URL for authorization.
    """
    base_url = 'https://hh.ru/oauth/authorize'
    response_type = 'code'
    oauth_url = f"{base_url}?response_type={response_type}&client_id={client_id}&redirect_uri={redirect_uri}"
    return oauth_url

# Application data
client_id = 'QCMBVLO34OTREB1U3ONRA40SFCA3FFCME70H0V5BKMPO7E3H0Q67N2VPPO4LIEPB'
redirect_uri = 'http://localhost'

# Generate OAuth URL
oauth_url = generate_oauth_url(client_id, redirect_uri)
print("URL for OAuth 2.0 authorization on HeadHunter:", oauth_url)

# Follow the URL printed above in a browser, log in, and authorize the application.
# You will be redirected to the specified redirect_uri with a "code" parameter in the URL.
# Copy that code and paste it below:

authorization_code = input("Enter the authorization code you received: ")

# Application data for token request
client_secret = 'IK96P3JB4U899VD1OO1A1ADM7NL3CC8H5OITN9IJDRENADEJ9M1L8LN9AV0DIGOS'
token_url = 'https://hh.ru/oauth/token'

# Parameters for token request
token_data = {
    'client_id': client_id,
    'client_secret': client_secret,
    'grant_type': 'authorization_code',
    'redirect_uri': redirect_uri,
    'code': authorization_code
}

# Sending POST request to get the token
response = requests.post(token_url, data=token_data)

# Checking the success of the token request
if response.status_code == 200:
    # Retrieve access token
    access_token = response.json()['access_token']
    print('Access Token:', access_token)
else:
    print('Error obtaining access token:', response.status_code, response.text)
