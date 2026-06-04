# Imports
import msal, requests, os
from bs4 import BeautifulSoup

# Constants
CLIENT_ID = "2cf97871-a8c5-4d24-9bec-1b809aca58bd"
TENANT_ID = "be4655df-ac73-401f-a7ae-198c3b72d0c6"
SCOPES = ["Mail.Read"]
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

# Functions
def update_count(count:dict):
    nombre_archivo = r"counter_file.txt"
    with open(nombre_archivo, "w", encoding = "utf-8") as c:
        for x in count:
            c.write(f"{x} {count[x]}\n")

def Ordenar(lista):
    count = {}
    with open("counter_file.txt", "r", encoding="utf-8") as p:
        for linea in p:
            if linea != "":
                clave, valor = linea.strip().split()
                valor = int(valor)
                count[clave] = valor
            else:
                break
    
    for x in lista:
        print(f"\nTexto: {x}")
        print("¿Que tipo de texto es?\n 0 = assignment\n 1 = class-teams\n 2 = club-activitie\n 3 = competition\n 4 = direct-mail\n 5 = exam-notice\n 6 = other")
        tipo = input("> ")

        while tipo not in ["0", "1", "2", "3", "4", "5", "6"]:
            tipo = input("Por favor, introduzca un valor correcto: ")
        
        match tipo:
            case "0":
                tipo = "assignment"
                carpeta = "Assignments"
            case "1":
                tipo = "class-teams"
                carpeta = "Classes-Teams"
            case "2":
                tipo = "club-activitie"
                carpeta = "Clubs-Activities"
            case "3":
                tipo = "competition"
                carpeta = "Competitions"
            case "4":
                tipo = "direct-mail"
                carpeta = "Direct-Mails"
            case "5":
                tipo = "exam-notice"
                carpeta = "Exam-Notices"
            case "6":
                tipo = "other"
                carpeta = "Others"
                
        if tipo not in count:
            count[tipo] = 1
        else:
            count[tipo] += 1
               
        carpeta_tipos = f"data/{carpeta}"

        nombre_archivo = f"{carpeta_tipos}/{tipo}-{count[tipo]}.txt"
        with open(nombre_archivo, "w", encoding= "utf-8") as t:
            t.write(x)

    update_count(count)

# Main
app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

flow = app.initiate_device_flow(scopes=SCOPES)
if "user_code" not in flow:
    raise SystemExit("Failed to start device flow: %s" % flow)

print(flow["message"])  

result = app.acquire_token_by_device_flow(flow)
if "access_token" not in result:
    raise SystemExit("Failed to obtain token: %s" % result)

headers = {"Authorization": "Bearer " + result["access_token"]}
url = "https://graph.microsoft.com/v1.0/me/messages?$top=50"
all_messages = []

while url:
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    all_messages.extend(data.get("value", []))
    url = data.get("@odata.nextLink")

print("Fetched", len(all_messages), "messages")
for m in all_messages:
    message_id = m["id"]
    detail_url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}"
    r = requests.get(detail_url, headers=headers)
    r.raise_for_status()
    detailed = r.json()
    body_html = detailed["body"]["content"]
    soup = BeautifulSoup(body_html, "html.parser")
    body_text = soup.get_text(separator="\n", strip=True)

    Ordenar([body_text])
    