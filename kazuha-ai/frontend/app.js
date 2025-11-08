const chat = document.getElementById("chat");
const textInput = document.getElementById("text");
const sendBtn = document.getElementById("send");
const talkBtn = document.getElementById("talk");

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "user" : "kazuha");
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  let emoji = "🙂";
  if (/!|angry|no|stupid|غضب|غاضب/i.test(text)) emoji = "😠";
  if (/
|wow|surpr/i.test(text)) emoji = "😲";
  avatar.textContent = emoji;
  el.appendChild(avatar);
  const bubble = document.createElement("span");
  bubble.className = "bubble";
  bubble.textContent = " " + text;
  el.appendChild(bubble);
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

sendBtn.onclick = async () => {
  const txt = textInput.value.trim();
  if (!txt) return;
  addMessage("user", txt);
  textInput.value = "";
  const r = await fetch("/chat_text", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text: txt})
  });
  const j = await r.json();
  if (j.error) {
    addMessage("kazuha", "Error: " + j.error);
  } else {
    addMessage("kazuha", j.text);
    if (j.audio_url) {
      const a = new Audio(j.audio_url);
      a.play();
    }
  }
};

let mediaRecorder;
let audioChunks = [];

talkBtn.onclick = async () => {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    if (!navigator.mediaDevices) {
      alert("Microphone not supported");
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, {type: 'audio/webm'});
      audioChunks = [];
      addMessage("user", "[voice message]");
      const fd = new FormData();
      fd.append("file", blob, "voice.webm");
      const res = await fetch("/chat_audio", {method: "POST", body: fd});
      const j = await res.json();
      if (j.error) {
        addMessage("kazuha", "Error: " + j.error);
      } else {
        addMessage("kazuha", j.text);
        if (j.audio_url) {
          const a = new Audio(j.audio_url);
          a.play();
        }
      }
    };
    mediaRecorder.start();
    talkBtn.textContent = "Stop";
  } else {
    mediaRecorder.stop();
    talkBtn.textContent = "Talk";
  }
};
