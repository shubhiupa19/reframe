import { NextResponse } from "next/server";
// define a function every time someone makes a POST reqest to /api/analyze endpoint

export async function POST(request) {
  // extracting the body of the request
  const body = await request.json();

  // specifically, pulls out the value from the key-val pair of "text" in the body
  const { text } = body;

  // then, we call the url of our python model
  const url = `${process.env.BACKEND_URL}/predict`;
  
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", },
    body: JSON.stringify({ text }),
  });

  const data = await response.json();

  // then, we get the result of the classification

  return NextResponse.json(data, {status: response.status});
}
