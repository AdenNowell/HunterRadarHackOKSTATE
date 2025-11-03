# HunterRadarHackOKSTATE
Hack OKSTATE Champion code and app from replit. 1st place best use of solana for this project. 24H
## Inspiration
Our inspiration is individuals whose lives or impact on hunting have been ruined due to the lack of safety measures on public land. Each year there are over 1,000 hunting related incidents on public land. That is 1,000 families who have lost a loved one, 1,000 hunters who have lost a love for the game, 1,000 preventable innocent mistakes. This app and device solves that problem. 
##What it does
Hunter Radar is a smart safety assistant that combines AI and blockchain to protect hunters in real time.
 - Detects nearby hunters and alerts users to prevent accidents.
 - Uses Gemini AI to provide safety coaching and regulation guidance.
 - Uses Solana blockchain to log consent and activity transparently.
 - ElevenLabs sound integration delivers spoken alerts and safety instructions even when visibility or connectivity is low.
Together, these systems ensure every signal, alert, and decision is traceable and safe.
## How we built it
We used a FastAPI backend hosted on Replit, connecting to multiple APIs:
 - Gemini API for retrieval-augmented safety insights.
 - Solana API for tamper-proof logging of user consent and device activity.
 - ElevenLabs API for converting AI responses into natural-sounding voice alerts.
 - The frontend runs a simple HTML/JS interface showing location-based alerts and status updates from connected devices.
## Challenges we ran into
 - Integrating multiple APIs smoothly under hackathon time constraints.
 - Port management, 8000 port and 5000 port did not communicate 
 - Managing real-time location and blockchain logging without slowing performance.
 - Designing a UI that is clear and safe to use outdoors, even on mobile.
 - Keeping data privacy strong while allowing transparency.
## Accomplishments that we're proud of
First ever Hackathon
Learned a ton about coding and API integration
Built a multi-layer AI safety system with blockchain logging and real-time sound alerts.
Successfully integrated ElevenLabs voice synthesis for natural audio safety feedback.
Proved that AI + blockchain + voice can make outdoor safety practical and ethical.
Created a working demo ready for field testing.
## What we learned
 - The uses of API integration into code. 
 - How to combine machine learning, blockchain, and human-centered design effectively.
 - The value of multi-sensory feedback. Adding sound dramatically improves safety response time.
 - How critical data transparency is in any AI-powered system.
## What's next for Hunter Radar
- Adding a LORA module into the port
 - Add Bluetooth wearable devices for live proximity and direction alerts.
 - Develop offline voice packs using ElevenLabs for remote areas.
 - Expand into other outdoor safety domains (fishing, hiking, rescue ops).
 - Partner with wildlife agencies and universities for pilot programs.
