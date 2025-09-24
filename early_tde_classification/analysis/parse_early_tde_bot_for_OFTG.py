from slack_sdk import WebClient
import datetime
import csv
import time
from slack_sdk.errors import SlackApiError

# Bot identification, and initialise client
early_tde_slack_token='xoxb-594392811169-9225778424023-dOL0ZUbZgXFvzUsxDQPfNw0w'
channel_id='C08LKCFSM7Z'
client = WebClient(token=early_tde_slack_token)

# Functions


def parse_msg(msg):
    """Parse message from Slack bot to extract relevant info for OFTG

    Args:
        msg (dict): message from Slack bot.

    Returns:
        dict: Dictionary including ZTF name, date of last alert, RA and Dec.
    """

    try:
        # Get name
        name = msg['text']

        # Get date and time of the last alert
        date = msg['blocks'][0]['fields'][1]['text'].split('*')[1]

        # Get radec
        coords = msg['blocks'][0]['fields'][2]['text'].split('*')[2][2:-1]
        ra = ' '.join(coords.split()[:3])
        dec = ' '.join(coords.split()[3:])

        return {
            "ztf_name": name,
            "date": date,
            "ra": ra,
            "dec": dec,
            }
    except:
        print("Unexpected error when parsing message. To verify")
        return None


def fetch_messages_limited():
    """Fetch last 15 messages given in Slack bot, and extract relevant information.
    
    Slack API only provides 15 messages even if the limit set is more.
    To get more we need the function below that will request iteratively to get more, 
    but might take very long if Slack blocks the bot requests.

    Returns:
        list of dicts: List where each element is a dict with extracted info from each message.
    """
    result = client.conversations_history(channel=channel_id, limit=20) 
            #, oldest = str(datetime.datetime(2025, 6, 1))  # -> Argument to add if you want to only get messages after certain date.
    parsed_data = []

    for msg in result.get("messages", []):
        if "bot_id" in msg and msg['bot_id'] == 'B08MADZMCP3':
            parsed = parse_msg(msg)
            if parsed:
                parsed_data.append(parsed)
    return parsed_data


def fetch_messages():
    """Fetch last messages given in Slack bot and extract relevant information.

    Returns:
        list of dicts: List where each element is a dict with extracted info from each message.
    """

    parsed_data = []
    cursor = None

    while True:
        try:
            response = client.conversations_history(
                channel=channel_id,
                limit=20,  
                cursor=cursor,
                # Argument to filter out messages before certain date.
                oldest = str(datetime.datetime(2025, 7, 25).timestamp()) # TODO: Put this as an argument to the function instead !
            )

            for msg in response.get("messages", []):
                if "bot_id" in msg and msg['bot_id'] == 'B08MADZMCP3':
                    parsed = parse_msg(msg)
                    if parsed:
                        parsed_data.append(parsed)

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            # Else sleep to allow API to recover ⏳
            time.sleep(1) 
        # If after sleeping API still does not respond, sleep as much as the API recommends.
        except SlackApiError as e:
            if e.response["error"] == "ratelimited":
                # Extract the recommended retry delay (default to 30s if not available)
                retry_after = int(e.response.headers.get("Retry-After", 30))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
    return parsed_data

def save_to_csv(data, filename="early_tde_candidates.csv"):
    """Save parsed data to csv file

    Args:
        data (list of dicts): output from fetch_messages
        filename (str, optional): csv file to save to. Defaults to "early_tde_candidates.csv".
    """
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ztf_name", "date", "ra", "dec"])
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    messages = fetch_messages()
    if messages:
        save_to_csv(messages)
        print(f"Saved {len(messages)} entries to CSV.")
    else:
        print("No matching messages found.")
