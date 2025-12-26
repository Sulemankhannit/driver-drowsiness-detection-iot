#  Real-Time Driver Drowsiness & Distraction Detection System

**A safety system powered by IoT and Computer Vision to prevent road accidents caused by fatigue.**

##  Project Overview
According to the MoRTH 2022 report, nearly 4,61,312 road accidents occurred in India, with driver fatigue being a major contributor. This project is a low-cost, real-time solution implemented on a **Raspberry Pi 4**. It monitors the driver's facial features to detect drowsiness, yawning, and distraction, utilizing a two-stage alert system (Local Alarm + Cloud Notification).

##  Features
* **Drowsiness Detection:** Calculates Eye Aspect Ratio (EAR) to detect closed eyes.
* **Yawn Detection:** Monitors Mouth Aspect Ratio (MAR) to identify fatigue.
* **Distraction Detection:** Triggers alert if the face is not visible for a set duration.
* **Driver Authentication:** Uses facial recognition to identify authorized drivers vs. guests.
* **IoT Alerts:**
    * *Stage 1:* Local Buzzer/LED + Push Notification (Pushover API).
    * *Stage 2:* Automated Emergency Call (Twilio API) + GPS Location SMS if the driver is unresponsive for 20s.

##  Tech Stack
* **Hardware:** Raspberry Pi 4, Pi Camera Module (5MP), Active Buzzer, LED.
* **Languages:** Python 3.11
* **Libraries:** OpenCV, Dlib, Face_recognition, Imutils, RPi.GPIO.
* **Cloud Services:** Twilio (Calls), Pushover (Notifications).

##  How It Works
1.  **Input:** The Pi Camera captures a live video stream.
2.  **Processing:** * Dlib's 68-point landmark predictor maps the face.
    * EAR and MAR are calculated in real-time.
3.  **Decision:** * If `EAR < 0.25` for 10 frames → **Drowsiness Alert**.
    * If `MAR > 0.5` for 15 frames → **Yawn Alert**.
4.  **Action:** Triggers GPIO pins for the buzzer and sends API requests for phone alerts.

##  Prerequisites (Hardware)
* Raspberry Pi 4 (4GB+ recommended)
* Pi Camera Module (5MP)
* Active Speaker & LED
* Internet connection (for API alerts)

##  Setup & Installation
1.  Clone the repository:
    ```bash
    git clone (https://github.com/Sulemankhannit/driver-drowsiness-detection-iot.git)
    ```
2.  Install dependencies:
    ```bash
   pip install -r requirements.txt
    ```
   Note: dlib installation on Raspberry Pi may take significant time (1-2 hours) to compile.

3. Download the Predictor Model: Download the shape_predictor_68_face_landmarks.dat file (approx 100MB) from        dlib.net, extract it, and place it in the project root folder. Also add sample Driver images in the known_faces folder.

4.  Add your API Keys in `main.py`.
    Open main.py and update the Cloud Service Credentials section with your own keys:
    Pushover User Key & API Token
    Twilio Account SID & Auth Token
    Run the system:
    ```bash
    python main.py
    ```
    Press q to quit the program.

##  Contributors
* **Suleman Khan** (Lead Developer)
* Raghav Dolyar
* Pranav Patidar
* Yogishwer Kumar
* Harkesh