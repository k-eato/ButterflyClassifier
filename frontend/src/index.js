import React from "react";
import { render } from 'react-dom';
import { ThemeProvider } from "@chakra-ui/core";

import Files from "./components/Files";

function App() {
  return (
    <ThemeProvider>
      <Files />
    </ThemeProvider>
  )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)
