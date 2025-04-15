'use client';
import Image from "next/image";
import { useCallback, useState } from "react";

export default function Home() {
  const [review, setReview] = useState('')
  const [response, setResponse] = useState('')
  const [inProgress, setInProgress] = useState(false);

  const callApi = useCallback(() => {
      setInProgress(true);
      const myHeaders = new Headers();
      myHeaders.append("Content-Type", "application/json");
      fetch('https://python-rag-review-analyser.onrender.com/query_review/', {
        method: "POST",
        body: JSON.stringify({text: review, k: 3}),
        headers: myHeaders
      }).then(res => res.json()).then(res => {
        setResponse(res.response)
      }).catch(_ => {
        setResponse("Something went wrong!!")
      }).finally( () => {
        setInProgress(false);
      }) 
  },[review])
  return (
    <div className="flex w-full h-screen items-center justify-center">
      <div className="flex min-w-100 items-center flex-col gap-5 justify-center border border-1 border-gray border-dashed p-10 rounded-lg">
      <input className="border border-1/2 pt-2 pb-2 pl-4 pr-4" placeholder="Enter a review" onChange={event => setReview(event.target.value)}/>
      {inProgress ?  <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-500 border-t-transparent"></div> : <button onClick={callApi} className="submit-btn">Analyse Review</button>}

      <div className="bg-gray-300 w-full max-w-100 h-auto p-10 rounded-lg text-left">
        Response: <pre>{JSON.stringify(response, null, 2)}</pre>
      </div>
    </div>
    </div>
  );
}
