import datareservoirio as drio
client_id = '51e67247-7da6-4f48-b5a1-b6adc0a5913a'
client_secret = 'RFj8Q~.LQLZzkO2im35AHE3W4la0djGyQ.APUcBn'

#client_id = os.environ.get('4subseaclientid')
#client_secret = os.environ.get('4subseaclientsecret')
if not client_id or not client_secret:
    raise ValueError("Missing client_id or client_secret environment variables.")
drio_auth = drio.authenticate.ClientAuthenticator(client_id=client_id, client_secret=client_secret)
drio_client = drio.Client(drio_auth)

df = drio_client.get(vega_lat, start=start_datetime,end=end_datetime)