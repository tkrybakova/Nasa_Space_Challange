// src/components/AnalyzeButton.jsx
import React, { useState } from "react";
import { Button, CircularProgress } from "@mui/material";
import { analyzeData, getResults } from "../api";

const AnalyzeButton = ({ fileId, onComplete }) => {
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const { task_id } = await analyzeData(fileId);

      // Пуллинг для проверки прогресса
      let result = null;
      while (!result || result.background_status !== "completed") {
        await new Promise((r) => setTimeout(r, 2000));
        result = await getResults(task_id);
      }

      onComplete(result);
    } catch (err) {
      console.error(err);
      alert("Ошибка анализа");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Button variant="contained" onClick={handleAnalyze} disabled={loading}>
      {loading ? <CircularProgress size={24} /> : "Начать анализ"}
    </Button>
  );
};

export default AnalyzeButton;
