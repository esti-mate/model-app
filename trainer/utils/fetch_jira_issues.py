import time
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os



JIRA_USER_EMAIL= "seniyas@lolctech.com"
JIRA_API_TOKEN = "ATATT3xFfGF0Es5pxnjOWZzwD9TF_tvVaLhQ6eLYqA8couGGosIbJdqtvj4vZFDDcPf3EODQ1tqluc337iizhO1xC4YwGkYWGzORQNw6BONDHRKVf0y_a6RMhQxlvGWSnZu_JJUhEFQxozOvVvVJ6G0tYQj1CUJS-THqBPW9Tkz7mu6sEDmo0mA=89706586"
auth = HTTPBasicAuth(JIRA_USER_EMAIL, JIRA_API_TOKEN)

JIRA_BASE_URL = 'https://lolcgroupdev.atlassian.net'
JIRA_API_URL = JIRA_BASE_URL + '/rest/api/3'
EXCEL_FILE_PATH = 'inputs/lolc-epics.xlsx'

# Function to make API call to get issue details
def get_issue_details(issue_id):
    url = f'{JIRA_API_URL}/issue/{issue_id}'
    response = requests.get(url,auth=auth)
    return response.json()

# Function to make API call to get attachment details
def get_attachment_details(attachment):
    id = attachment.get("id")
    url = f'{JIRA_API_URL}/attachment/content/{id}'
    response = requests.get(url,auth=auth)
    return response.content  # Assuming the response is the binary content of the attachment

# Function to save attachments to a folder
def save_attachment(issue_id, attachment, attachment_content):
    attach_id = attachment.get("id")
    file_name = attachment.get("filename")
    folder_path = f'attachments/{issue_id}'
    os.makedirs(folder_path, exist_ok=True)
    
    with open(os.path.join(folder_path, f'{file_name}'), 'wb') as f:
        f.write(attachment_content)



def startProcess():
    # Load Excel sheet into a DataFrame
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='sheet01')
    # Iterate through rows
    for index, row in df.iterrows():
        issue_id = row[1]  # Replace 'IssueId' with the actual column name in your Excel sheet

        if issue_id   :
            print(f'{issue_id}-------------------------------------------------------------------')
            issue_details = get_issue_details(issue_id)

            # Assuming 'AttachmentIds' is a key in the API response containing a list of attachment ids
            attachments = issue_details.get('fields', []).get('attachment', [])
            print(f' - attachment count : {len(attachments)}')

            for attachment in attachments:
                print(f'   - Downloading attachment {attachment.get("filename")}...')
                attachment_content = get_attachment_details(attachment)
                save_attachment(issue_id, attachment, attachment_content)
                time.sleep(1)
            time.sleep(2)  # Sleep for 1 second to avoid hitting the rate limit
    print("Attachments downloaded and saved successfully.\n")
    return {"message":"Attachments downloaded and saved successfully."}


def get_folder_names_with_pdf_or_word_files():
    attachment_folder = 'attachments'
    folder_names = []
    pdf_folderNames = []
    for folder_name in os.listdir(attachment_folder):
        folder_path = os.path.join(attachment_folder, folder_name)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith('.pdf') or file.endswith('.doc') or file.endswith('.docx'):
                    folder_names.append(folder_name)
                    if file.endswith('.pdf'):
                        pdf_folderNames.append(folder_name)
                    break
    print( str(len(folder_names)) + ".pdf or .doc or .docx files are available")
    print( "* "+str(len(pdf_folderNames)) + " pdf files are available:", pdf_folderNames)
    return folder_names

if __name__ == "__main__":
    # You can test the function here if needed
    result = get_folder_names_with_pdf_or_word_files()
    print(result)
