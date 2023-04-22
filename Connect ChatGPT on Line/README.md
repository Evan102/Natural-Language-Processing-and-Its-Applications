# Connect ChatGPT to LINE chatbot

## 1. Create your own LINE message API account
https://developers.line.biz/en/

reference: https://ithelp.ithome.com.tw/articles/10192928

## 2. Download ngrok to support "https"
https://ngrok.com/download

## 3. Run ngrok 
Open a terminal and cd to the directory where you put ngrok.exe. 

Command on the terminal: "ngrok http 5002", and keep the terminal on.

## 4. Copy an address to LINE Webhook URL 
After step 3, you will see an address which is "https:XXX.ngrok-free.app" behind the line  "Forwarding".

Please copy "https:XXX.ngrok-free.app" and paste on LINE Webhook URL. 

<img src="https://github.com/Evan102/Natural-Language-Processing-and-Its-Applications/blob/main/Connect%20ChatGPT%20on%20Line/Line%20Developers%20-%20Webhook%20URL.png"  width="60%" height="30%">

## 5. Run ChatGPT on Line bot
Update your own OpenAI API key, LINE chatbot Channel secret, and LINE chatbot Channel access token in ChatGPTonLINE.py.
<img src="https://github.com/Evan102/Natural-Language-Processing-and-Its-Applications/blob/main/Connect%20ChatGPT%20on%20Line/Line%20Developers%20-%20Channel%20secret.png"  width="100%" height="60%">

<img src="https://github.com/Evan102/Natural-Language-Processing-and-Its-Applications/blob/main/Connect%20ChatGPT%20on%20Line/Line%20Developers%20-%20Channel%20access%20token.png"  width="100%" height="60%">

Open another terminal and command: "python ChatGPTonLINE.py", and keep the terminal on.

After you run "ChatGPTonLINE.py", you can verify if your Webhook URL is successful and test your LINE chatbot.

