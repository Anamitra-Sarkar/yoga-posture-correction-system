import { Platform, StyleSheet } from "react-native";
import { WebView } from "react-native-webview";
import { useEffect, useRef } from "react";

import type { Landmark } from "@/lib/pose-geometry";

type DetectorRequest = { id: string; base64: string } | null;

const workerHtml = `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1" /></head><body><script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script><script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script><script>
let pose; let currentId; let ready=false;
function send(payload){ window.ReactNativeWebView.postMessage(JSON.stringify(payload)); }
function createPose(){
  pose = new Pose({locateFile:(file)=>'https://cdn.jsdelivr.net/npm/@mediapipe/pose/'+file});
  pose.setOptions({modelComplexity:1,smoothLandmarks:true,enableSegmentation:false,smoothSegmentation:false,minDetectionConfidence:0.6,minTrackingConfidence:0.6});
  pose.onResults((results)=>{
    const landmarks=(results.poseLandmarks||[]).map((point)=>({x:point.x,y:point.y,z:point.z,visibility:point.visibility||0}));
    send({type:'result',id:currentId,landmarks});
  });
  ready=true; send({type:'ready'});
}
window.__asanaProcessFrame=async function(encoded){
  try { const payload=JSON.parse(encoded); if(!ready) createPose(); currentId=payload.id; const image=new Image(); image.onload=async()=>{try{await pose.send({image});}catch(error){send({type:'error',id:payload.id,message:String(error)})}}; image.onerror=()=>send({type:'error',id:payload.id,message:'Unable to decode camera frame'}); image.src='data:image/jpeg;base64,'+payload.base64; }
  catch(error){ send({type:'error',message:String(error)}); }
};
createPose();
</script></body></html>`;

export function PoseDetectorWorker({ request, onReady, onResult, onError }: {
  request: DetectorRequest;
  onReady: () => void;
  onResult: (id: string, landmarks: Landmark[]) => void;
  onError: (message: string) => void;
}) {
  const ref = useRef<WebView>(null);

  useEffect(() => {
    if (!request || Platform.OS === "web") return;
    const encoded = JSON.stringify(request);
    ref.current?.injectJavaScript(`window.__asanaProcessFrame(${JSON.stringify(encoded)}); true;`);
  }, [request]);

  if (Platform.OS === "web") return null;
  return <WebView ref={ref} source={{ html: workerHtml }} onMessage={(event) => {
    try {
      const message = JSON.parse(event.nativeEvent.data);
      if (message.type === "ready") onReady();
      else if (message.type === "result") onResult(message.id, message.landmarks ?? []);
      else if (message.type === "error") onError(message.message ?? "Pose detector unavailable");
    } catch { onError("Pose detector returned an unreadable response"); }
  }} style={styles.worker} javaScriptEnabled originWhitelist={["*"]} />;
}

const styles = StyleSheet.create({ worker: { width: 1, height: 1, opacity: 0, position: "absolute", left: -10, top: -10 } });
