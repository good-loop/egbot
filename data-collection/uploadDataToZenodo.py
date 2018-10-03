import requests, os, sys, json

BASE_URL = "https://sandbox.zenodo.org" 
TOKEN = 'Vk6qHrTlpgmlUyfcEonKZgKUNbkFoQ79IZhOaLbweZJEO6KCEuHnfNmHbE9v'#'cujfEB48YOxOuy2CcT6FnEEi7I7fxiZCnQY4isfvGsK2DqqNgP5dpd2hZwgl'
description = "<p>This dataset was obtained from the <a href=\"https://stackexchange.com/\">StackExchange</a> website, by querying the <a href=\"https://api.stackexchange.com/\">Stack Exchange API</a> according to its documentation.</p><p>It consists of approximately 1 million mathematics questions and their respective answers, as well as markers of interaction quality (such as user-provided scoring of question and answer quality) and social dynamics (reputation scores, badges, etc).</p><p>This dataset was compiled for the purpose of doing text-based analysis of mathematical discourse and for the construction of a conversational math bot.</p><p>&nbsp;</p><p>&nbsp;</p>"
headers = {"Content-Type": "application/json"}

metadata = {
    'metadata': {
        'title': 'Mathematics Stack Exchange API Q&A Data',
        'upload_type': 'dataset',
        'description': description,
        'creators': [{'name': 'Preda, Irina'}],
        'version': '1.0',
        'access_right': 'open', 
        "license": { "id": "CC-BY-SA-4.0" }, 
        "communities": [{ "identifier": "egbot" }]
    }
}

def upload(data_file):
    data_path = os.path.join(directory, data_file)

    # Create new data submission
    url = "{base_url}/api/deposit/depositions".format(base_url=BASE_URL)
    headers = {"Content-Type": "application/json"}
    print("Preparing submission ...")
    response = requests.post(url, params={'access_token': TOKEN}, json=metadata, headers=headers)
    if response.status_code > 210:
        print("Error happened during submission, status code: " + str(response.status_code))
        print(response.text)
        return

    # Get the submission ID
    submission_id = json.loads(response.text)["id"]
    bucket_url = response.json()['links']['bucket']

    # Upload the file
    #url = "{base_url}/api/deposit/depositions/{id}/files".format(base_url=BASE_URL, id=str(submission_id))
    upload_metadata = {'filename': data_file}
    files = {'file': open(data_path, 'rb')}
    print("Uploading data ...")
    response = requests.put('%s/%s' % (bucket_url,upload_metadata),data=files,
                                    headers={"Accept":"application/json",
                                    "Authorization":"Bearer %s" % TOKEN,
                                    "Content-Type":"application/octet-stream"})
    #(url, params={'access_token': TOKEN}, data=upload_metadata, files=files)
    #print(response.text)
    if response.status_code > 210:
        print("Error happened during file upload, status code: " + str(response.status_code))
        return
    
    print("{file} submitted with submission ID = {id} (DOI: 10.5281/zenodo.{id})".format(file=data_path,id=submission_id))    
    # The submission needs an additional "Publish" step. This can also be done from a script, but to be on the safe side, it is not included. (The attached file cannot be changed after publication.)

def batch_upload(directory):
    for data_file in os.listdir(directory):
        data_path = os.path.join(directory, data_file)
        if data_path.endswith(".zip"):
            print("Uploading %s" % (data_path))
            upload(data_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: upload_to_zenodo.py <directory>")
        print("  The directory contains .json metadata and the .json data files.")
        exit()
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Invalid directory.")
        exit()
   
batch_upload(directory)

