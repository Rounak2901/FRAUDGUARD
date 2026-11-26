import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import LiveMonitor from "./pages/LiveMonitor";
import TransactionDetail from "./pages/TransactionDetail";
import ReviewQueue from "./pages/ReviewQueue";
import EntityRisk from "./pages/EntityRisk";
import ModelInsights from "./pages/ModelInsights";
import CostSimulator from "./pages/CostSimulator";
import TryIt from "./pages/TryIt";
import About from "./pages/About";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<LiveMonitor />} />
        <Route path="/tx/:id" element={<TransactionDetail />} />
        <Route path="/review" element={<ReviewQueue />} />
        <Route path="/entities" element={<EntityRisk />} />
        <Route path="/insights" element={<ModelInsights />} />
        <Route path="/cost" element={<CostSimulator />} />
        <Route path="/try" element={<TryIt />} />
        <Route path="/about" element={<About />} />
      </Route>
    </Routes>
  );
}
