# Importuojamos reikalingos bibliotekos
import os
import cv2
import numpy as np
import time
import ftplib
from threading import Thread
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from fastapi import FastAPI, Request, File, UploadFile, Response
from fastapi.responses import StreamingResponse
import nest_asyncio
import uvicorn
import asyncio
from anyio.to_thread import run_sync
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
# from tflite_runtime.interpreter import Interpreter
import tensorflow as tf
from queue import Queue
import os
from dotenv import load_dotenv
# Sukuriamas FastAPI objektas
saitynas = FastAPI()

# Nustatome CORS politiką
saityno_pasiekiamumas = [
    "*",
]

saitynas.add_middleware(
    CORSMiddleware,
    allow_origins=saityno_pasiekiamumas,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv('.env')

# FTP ir el. pašto prisijungimo duomenys
ftp_adresas = os.getenv('FTP_ADRESAS')
ftp_portas = int(os.getenv('FTP_PORTAS'))
ftp_vartotojas = os.getenv('FTP_VARTOTOJAS')
ftp_slaptazodis = os.getenv('FTP_SLAPTAZODIS')
gmail_vartotojas = os.getenv('GMAIL_VARTOTOJAS')
gmail_slaptazodis = os.getenv('GMAIL_SLAPTAZODIS')

# Kelias į saityno nuotraukos aplankalą
nuotolinis_kelias = '/public_html/nuotraukos'

# Failo įkėlimo funkcija į saityno failus
def failo_kelimo_ftp_funkcija(vietinis_failas, nuotolinis_failas):
    ftp = ftplib.FTP()
    ftp.connect(ftp_adresas, ftp_portas)
    ftp.login(ftp_vartotojas, ftp_slaptazodis)
    ftp.cwd(nuotolinis_kelias)
    with open(vietinis_failas, 'rb') as failas:
        ftp.storbinary(f'STOR {nuotolinis_failas}', failas)
    ftp.quit()

# Laiško siuntimo funkcija
def laisko_siuntimo_funkcija(nuotraukos_kelias):
    gavejas = 'vartotojas@gmail.com'
    tema = 'Ugnis aptikta'
    zinute = MIMEMultipart()
    svetaines_email = 'info@forestcam.eu'
    zinute['From'] = svetaines_email
    zinute['To'] = gavejas
    zinute['Subject'] = tema
    tekstas = 'Ugnis aptikta, prideta nuotrauka ugnies.'
    zinute.attach(MIMEText(tekstas))
    with open(nuotraukos_kelias, 'rb') as f:
        nuotraukos_info = f.read()
    nuotrauka = MIMEImage(nuotraukos_info)
    zinute.attach(nuotrauka)
    try:
        serveris = smtplib.SMTP('smtp.gmail.com', 587)
        serveris.starttls()
        serveris.login(gmail_vartotojas, gmail_slaptazodis)
        tekstas = zinute.as_string()
        serveris.sendmail(svetaines_email, gavejas, tekstas)
        serveris.quit()
        print('Laiskas isiustas sekmingai!')
    except Exception as e:
        print(f'Nepavyko isiusti laisko: {e}')

# Sukuriame klasę Vaizdo_klase, kuri skirta vaizdo transliacijai valdyti
class Vaizdo_klase:
    # Inicializavimo funkcija nustato vaizdo transliacijos parametrus
    def __init__(self,skiriamoji_geba=(640,480),kadrai=10):
        # Nustatomas vaizdo transliacijos šaltinis (kamerą)
        self.srautas = cv2.VideoCapture(0)
        # Nustatomas kodavimo formatas
        ret = self.srautas.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # Nustatoma vaizdo skiriamoji_geba
        ret = self.srautas.set(3,skiriamoji_geba[0])
        ret = self.srautas.set(4,skiriamoji_geba[1])
        # Fiksuojamas ir skaitomas pirmas vaizdo kadras
        (self.pagauti, self.kadras) = self.srautas.read()
        # Nustatome ar transliacija sustabdyta
        self.sustabdyti = False

    # Funkcija start() paleidžia atnaujinimo procesą atskiroje gijose
    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    # Funkcija update() nuolat atnaujina vaizdo kadro informaciją
    def update(self):
        while True:
            if self.sustabdyti:
                self.srautas.release()
                return
            (self.pagauti, self.kadras) = self.srautas.read()

    # Funkcija read() grąžina paskutinį vaizdo kadrą
    def read(self):
        return self.kadras

    # Funkcija stop() sustabdo vaizdo transliaciją
    def stop(self):
        self.sustabdyti = True

# For Raspberry Pi

# Nustatome modelio, objektų aptikimo failo ir etikečių failo kelius
# modelio_vieta = '/home/pi/Desktop/Forestcam/ssd'
# objektu_aptikimo_failas = '/home/pi/Desktop/Forestcam/ssd/model.tflite'
# etikeciu_failas = '/home/pi/Desktop/Forestcam/ssd/labelmap.txt'

# For Windows

modelio_vieta = 'path_to_model'
objektu_aptikimo_failas = 'path_to_tflite'
etikeciu_failas = 'path_to_labelmap.txt'


# Minimalus aptikimo slenkstis
min_aptikimo_slenkstis2= 0.3
min_aptikimo_slenkstis= 0.7
# Skiriamoji_geba
resW, resH = '800x540'.split('x')
imW, imH = int(resW), int(resH)

# Įkeliame failus
grazinti_failu_kelia = os.getcwd()
obj_aptk_f_kelio_sujung = os.path.join(grazinti_failu_kelia, modelio_vieta, objektu_aptikimo_failas)
etk_f_kelio_sujung = os.path.join(grazinti_failu_kelia, modelio_vieta, etikeciu_failas)

# Nuskaitome etiketes iš failo
with open(etk_f_kelio_sujung, 'r') as f:
    etiketes = [line.strip() for line in f.readlines()]

# Jeigu pirmoji etiketė yra neaiški, pašaliname
if etiketes[0] == '???':
    del(etiketes[0])

vertejas = tf.lite.Interpreter(model_path=obj_aptk_f_kelio_sujung)

# Alokavimas vietos modelio tenzorams
vertejas.allocate_tensors()

# Gauname informaciją apie modelio įvesties ir išvesties tenzorus
ivesties_informacija = vertejas.get_input_details()
isvesties_informacija = vertejas.get_output_details()

# Išskleidžiame įvesties tenzoriaus formą, gauname jo plotį ir aukštį
aukstis = ivesties_informacija[0]['shape'][1]
plotis = ivesties_informacija[0]['shape'][2]

# Nustatome ar modelis naudoja slankųjį kablelį
slankusis_modelis = (ivesties_informacija[0]['dtype'] == np.float32)

# Nustatome įvesties vidurkį ir standartinį nuokrypį
ivesties_vidurkis = 127.5
ivesties_std = 127.5 

# Gauname išvesties sluoksnio pavadinimą
isvesties_sluoksnio_pavadinimas = isvesties_informacija[0]['name']

# Sukuriame indeksus, kuriuos naudosime išvesties tenzoriuose
aptikimo_linijos_idx, aptikimo_klasiu_idx, sutapciu_idx = 1, 3, 0

# Inicijuojame vaizdo transliaciją su nurodyta skiriamoji_geba ir kadrų skaičiumi
Vaizdo_klase = Vaizdo_klase(skiriamoji_geba=(imW, imH), kadrai=30).start()
# Palaukime, kol kamera pradės transliaciją
time.sleep(1)

# Priskiriame kintamuosius laiko žymoms saugoti
paskutinio_el_pasto_siuntimo_laikas = 0
paskutinio_ikelimo_laikas = {}

# Sukuriame etikečių spalvų sąrašą
np.random.seed(10)
etikeciu_spalvos = np.random.randint(100, 255, size=(len(etiketes),3)).tolist()

# Objektų aptikimo funkcija, kuri analizuoja kadrą ir grąžina jį su aptiktų objektų rėmeliais ir etiketėmis
def objektu_aptikimo_funckija(nuotrauka, imW, imH):
    desired_object_detected = False
    # Konvertuojame nuotrauką į RGB spalvų erdvę
    kadro_rgb = cv2.cvtColor(nuotrauka, cv2.COLOR_BGR2RGB)
    # Keičiame kadro dydį, kad atitiktų modelio įvesties dydį
    kadro_redagavimas = cv2.resize(kadro_rgb, (plotis, aukstis))
    # Pridedame papildomą dimensiją, kad atitiktų modelio įvesties formą
    kadro_informacija = np.expand_dims(kadro_redagavimas, axis=0)
    # Jei modelis naudoja slankųjį kablelį, normalizuojame kadro informaciją
    if slankusis_modelis:
        kadro_informacija = (np.float32(kadro_informacija) - ivesties_vidurkis) / ivesties_std
    # Patalpiname kadro informaciją į modelio įvesties tenzorių
    vertejas.set_tensor(ivesties_informacija[0]['index'], kadro_informacija)
    # Modelio iškvietimas
    vertejas.invoke()
    # Gauname modelio išvesties tenzorių reikšmes
    aptikimo_linijos = vertejas.get_tensor(isvesties_informacija[aptikimo_linijos_idx]['index'])[0]
    aptikimo_klasiu_indeksas = vertejas.get_tensor(isvesties_informacija[aptikimo_klasiu_idx]['index'])[0]
    sutapciu_rezultatas = vertejas.get_tensor(isvesties_informacija[sutapciu_idx]['index'])[0]
    # Iteruojame per aptiktus objektus
    for i in range(len(sutapciu_rezultatas)):
        # Tikriname ar aptikimo rezultatas yra tarp nustatyto slenksčio ir 1.0
        if ((sutapciu_rezultatas[i] > min_aptikimo_slenkstis2) and (sutapciu_rezultatas[i] <= 1.0)):
            # Gauname rėmelio koordinates ir apskaičiuojame jas pagal nuotraukos dydį
            minimalus_y = int(max(1, (aptikimo_linijos[i][0] * imH)))
            minimalus_x = int(max(1, (aptikimo_linijos[i][1] * imW)))
            maksimalus_y = int(min(imH, (aptikimo_linijos[i][2] * imH)))
            maksimalus_x = int(min(imW, (aptikimo_linijos[i][3] * imW)))
            # Gauname spalvą pagal etiketės indeksą
            spalva = tuple(etikeciu_spalvos[int(aptikimo_klasiu_indeksas[i])])
            # Piešiame rėmelį aplink aptiktą objektą           
            cv2.rectangle(nuotrauka, (minimalus_x, minimalus_y), (maksimalus_x, maksimalus_y), spalva, 2)
            # Gauname objekto pavadinimą ir etiketę su tikimybe
            objekto_pavadinimas = etiketes[int(aptikimo_klasiu_indeksas[i])]
            etikete = '%s: %d%%' % (objekto_pavadinimas, int(sutapciu_rezultatas[i] * 100))
            # Gauname etiketės dydį ir pradinę liniją
            etikes_dydis, pradine_linija = cv2.getTextSize(etikete, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Apskaičiuojame etiketės minimalų y koordinatę
            etiketes_minimalus_y = max(minimalus_y, etikes_dydis[1] + 10)
            if objekto_pavadinimas in ["Briedis", "Kiskis", "Lape", "Sernas", "Stirna", "Vilkas", "Ugnis"]:
                # Piešiame stačiakampį etiketės fono spalva
                cv2.rectangle(nuotrauka, (minimalus_x, etiketes_minimalus_y - etikes_dydis[1] - 10), (minimalus_x + etikes_dydis[0], etiketes_minimalus_y + pradine_linija - 10), spalva, cv2.FILLED)
                # Rašome etiketę ant nuotraukos
                cv2.putText(nuotrauka, etikete, (minimalus_x, etiketes_minimalus_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                desired_object_detected = True  # Set the flag to True if a desired object is detected

        if desired_object_detected:
            return nuotrauka
        else:
            return None
        
# Saityno tarnybinė, kuri įkelia nuotrauką, atlika objektų aptikimą ir grąžina rezultatą
@saitynas.post("/kelti-nuotraukas/")
async def nuotraukos_ikelimo_funkcija(nuotrauka: UploadFile = File(...)):
    # Skaitome įkeltą nuotrauką kaip dvejetainę informaciją
    failo_dvejataine_informacija = await nuotrauka.read()
    # Konvertuojame dvejetainę informaciją į NumPy masyvą ir dekoduojame nuotrauką naudojant OpenCV
    np_nuotrauka = cv2.imdecode(np.frombuffer(failo_dvejataine_informacija, np.uint8), -1)
    # Gauname nuotraukos aukštį, plotį ir gylį
    nuotraukos_aukstis, nuotraukos_plotis, _ = np_nuotrauka.shape
    # Atliekame objektų aptikimą įkeltai nuotraukai
    aptikti_nuotrauka = objektu_aptikimo_funckija(np_nuotrauka, nuotraukos_plotis, nuotraukos_aukstis)
    # Check if the returned value is not None
    if aptikti_nuotrauka is not None:
        # Konvertuojame aptiktą nuotrauką į JPEG formatą
        _, buferis = cv2.imencode('.jpg', aptikti_nuotrauka)
        # Konvertuojame JPEG buferį į dvejetainę informaciją
        nuotraukos_dvejataine_informacija = buferis.tobytes()
        # Grąžiname nuotrauką su aptiktais objektais kaip "StreamingResponse", kad galima būtų peržiūrėti saityne
        return StreamingResponse(BytesIO(nuotraukos_dvejataine_informacija), media_type="image/jpeg")
    return Response(status_code=204)  # Return an empty response with 204 No Content status code

# Apibrėžiame pagrindinio maršruto funkcija, kuri grąžina vaizdo transliaciją
@saitynas.get('/')
async def index(request: Request):
    return StreamingResponse(saityno_transliacija(), media_type = "multipart/x-mixed-replace;boundary=kadras")

# Inicializuojama eilė kadro informacijai laikyti
kadro_eile = Queue(maxsize = 3)

# Kadro atnaujinimo funkcija, kuri stebi ir atnaujina kadro eilę
async def kadro_atnaujinimo_funkcija():
    while True:
        if not kadro_eile.full():
            kadru_informacija = Vaizdo_klase.read()
            kadro_eile.put(kadru_informacija)
        await asyncio.sleep(0.001)

# Užtikrinama, kad kadro atnaujinimo funkcija bus vykdoma asyncio cikle
asyncio.ensure_future(kadro_atnaujinimo_funkcija())

# Saityno transliacijos funkcija
async def saityno_transliacija():
    global paskutinio_el_pasto_siuntimo_laikas
    global paskutinio_ikelimo_laikas

    # Pagrindinis transliacijos ciklas
    while True:
        # Nuskaitome kadro informacija
        kadru_informacija = Vaizdo_klase.read()
        # Kopijuojame kadrą
        kadras = kadru_informacija.copy()

        """
        Veikimo principas toks pat kaip ir aprašytame (def objektu_aptikimo_funkcija)
        """
        kadro_rgb = cv2.cvtColor(kadras, cv2.COLOR_BGR2RGB)
        kadro_redagavimas = cv2.resize(kadro_rgb, (plotis, aukstis))
        kadro_informacija = np.expand_dims(kadro_redagavimas, axis=0)

        if slankusis_modelis:
            kadro_informacija = (np.float32(kadro_informacija) - ivesties_vidurkis) / ivesties_std

        vertejas.set_tensor(ivesties_informacija[0]['index'], kadro_informacija)
        vertejas.invoke()

        aptikimo_linijos = vertejas.get_tensor(isvesties_informacija[aptikimo_linijos_idx]['index'])[0]
        aptikimo_klasiu_indeksas = vertejas.get_tensor(isvesties_informacija[aptikimo_klasiu_idx]['index'])[0]
        sutapciu_rezultatas = vertejas.get_tensor(isvesties_informacija[sutapciu_idx]['index'])[0]

        for i in range(len(sutapciu_rezultatas)):
            if ((sutapciu_rezultatas[i] > min_aptikimo_slenkstis) and (sutapciu_rezultatas[i] <= 1.0)):
                minimalus_y = int(max(1, (aptikimo_linijos[i][0] * imH)))
                minimalus_x = int(max(1, (aptikimo_linijos[i][1] * imW)))
                maksimalus_y = int(min(imH, (aptikimo_linijos[i][2] * imH)))
                maksimalus_x = int(min(imW, (aptikimo_linijos[i][3] * imW)))

                spalva = tuple(etikeciu_spalvos[int(aptikimo_klasiu_indeksas[i])])
                cv2.rectangle(kadras, (minimalus_x, minimalus_y), (maksimalus_x, maksimalus_y), spalva, 2)

                objekto_pavadinimas = etiketes[int(aptikimo_klasiu_indeksas[i])]
                etikete = '%s: %d%%' % (objekto_pavadinimas, int(sutapciu_rezultatas[i] * 100))
                etikes_dydis, pradine_linija = cv2.getTextSize(etikete, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                etiketes_minimalus_y = max(minimalus_y, etikes_dydis[1] + 10)
                cv2.rectangle(kadras, (minimalus_x, etiketes_minimalus_y - etikes_dydis[1] - 10), (minimalus_x + etikes_dydis[0], etiketes_minimalus_y + pradine_linija - 10), spalva, cv2.FILLED)
                cv2.putText(kadras, etikete, (minimalus_x, etiketes_minimalus_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


                # Jeigu praėjo daugiau nei 60s ir objektas yra ugnis, tai nuotrauka yra įkeliama į saityno aplankalą ir siunčiamas el. laiškas su objektu. 
                if objekto_pavadinimas == "Ugnis" and (time.time() - paskutinio_el_pasto_siuntimo_laikas) > 60:
                   # Įrašome kadrą su aptikta ugnimi į failą "ugnis_aptikta.jpg"
                    cv2.imwrite("ugnis_aptikta.jpg", kadras)
                    # Siunčiame el. laišką su ugnies nuotrauka
                    laisko_siuntimo_funkcija("ugnis_aptikta.jpg")
                    # Keliame failą su ugnies nuotrauka į saityno aplankalą
                    failo_kelimo_ftp_funkcija("ugnis_aptikta.jpg", "ugnis_aptikta.jpg")
                    # Atnaujiname paskutinio el. laiško siuntimo laiką
                    paskutinio_el_pasto_siuntimo_laikas = time.time()

                """
                Tikrina ar aptiktas objektas nebuvo įrašytas paskutinio įkėlimo laikų arba praėjo
                daugiau nei 10 sekundžių nuo paskutinio to paties objekto įkėlimo į saityno folderį
                """
                if objekto_pavadinimas in ["Briedis", "Kiskis", "Lape", "Sernas", "Stirna", "Vilkas", "Ugnis"]:
                    if objekto_pavadinimas not in paskutinio_ikelimo_laikas or (time.time() - paskutinio_ikelimo_laikas[objekto_pavadinimas]) > 10:
                        screenshot_name = f'{objekto_pavadinimas}_{int(time.time())}_aptikta.jpg'
                        cv2.imwrite(screenshot_name, kadras)
                         # Keliame failą su objekto nuotrauką į saityno aplankalą
                        failo_kelimo_ftp_funkcija(screenshot_name, screenshot_name)
                        # Atnaujiname paskutinio objekto įkėlimo laiką
                        paskutinio_ikelimo_laikas[objekto_pavadinimas] = time.time()

        _, buferis = cv2.imencode('.jpg', kadras)
        # Konvertuojame kadrą į baitų seką
        kadras = buferis.tobytes()

        # Generuojame kadrą su MIME antrašte ir nuotraukos tipu, kad jį galėtume naudoti vaizdo transliacijai
        yield (b'--kadras\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + kadras + b'\r\n')
    
                # Pauzė tarp aptikimo ciklų
        await asyncio.sleep(0.001)

if __name__ == '__main__':
    try:
        # Taikome "nest_asyncio", kad galima būtų naudoti "asyncio" kartu su "jupyter"
        nest_asyncio.apply()
        # Paleidžiame "uvicorn" saityno tarnybinę, kuri klausosi 0.0.0.0 IP adreso ir 8000 prievado
        uvicorn.run(saitynas, host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        # Jei gauname klavišų paspaudimą (pvz., Ctrl+C), sustabdome vaizdo transliaciją ir uždarome programą
        asyncio.get_event_loop().run_until_complete(run_sync(Vaizdo_klase.stop))