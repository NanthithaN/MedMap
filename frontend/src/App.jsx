import React, { useState, useEffect } from 'react';
import { Network, Search, Database, Stethoscope, CheckCircle, AlertCircle } from 'lucide-react';

const API_URL = "http://localhost:8000";

export default function App() {
  const [areas, setAreas] = useState([]);
  const [selectedAreas, setSelectedAreas] = useState([]);
  const [capacity, setCapacity] = useState(500);
  const [loading, setLoading] = useState(false);
  const [trainingLoader, setTrainingLoader] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Initial load
  useEffect(() => {
    fetch(`${API_URL}/areas`)
      .then(r => r.json())
      .then(d => {
        setAreas(d.areas || []);
      })
      .catch(e => console.error("Could not load areas", e));
  }, []);

  const handleTrainModel = async () => {
    setTrainingLoader(true);
    setTrainingStatus("Training XGBoost on 50,000 areas...");
    try {
      const res = await fetch(`${API_URL}/train-model`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail);
      setTrainingStatus(`Success! Global Model Trained. (MSE: ${data.mse?.toFixed(2) || 'N/A'})`);
    } catch (e) {
      setTrainingStatus(`Error: ${e.message}`);
    }
    setTrainingLoader(false);
  };

  const handlePredictAndOptimize = async (e) => {
    e.preventDefault();
    if (selectedAreas.length === 0) {
      setError("Please select at least one area.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // 1. Predict Demand
      const predictRes = await fetch(`${API_URL}/predict-demand`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_areas: selectedAreas })
      });
      const predictData = await predictRes.json();
      
      if (!predictRes.ok) throw new Error(predictData.detail || "Failed to predict demand");

      const predicted_demands = {};
      Object.keys(predictData.predictions).forEach(area => {
        predicted_demands[area] = predictData.predictions[area].demand;
      });

      // 2. Optimize
      const optRes = await fetch(`${API_URL}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_areas: selectedAreas,
          predicted_demands: predicted_demands,
          capacity: Number(capacity)
        })
      });
      const optData = await optRes.json();
      
      if (!optRes.ok) throw new Error(optData.detail || "Optimization failed inside MILP");

      setResults({
        demands: predictData.predictions,
        optimization: optData
      });

    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  const handleSelectArea = (e) => {
    const value = Array.from(e.target.selectedOptions, option => option.value);
    setSelectedAreas(value);
  }

  return (
    <div className="app-container">
      <header>
        <h1>MedMap Intelligence</h1>
        <p>Optimizing Healthcare Facility Placement via Machine Learning & MILP</p>
      </header>

      {/* Global Model Control */}
      <div className="glass-card mb-6" style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: '#c084fc' }}>
            <Database size={20} /> Machine Learning Engine
          </h3>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Train the XGBoost regressor globally on all 50k areas to predict future demand.</p>
        </div>
        <div style={{ textAlign: 'right' }}>
          <button onClick={handleTrainModel} disabled={trainingLoader} className="btn-secondary">
            {trainingLoader ? <span className="loader"></span> : "Train Global Model"}
          </button>
          {trainingStatus && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: trainingStatus.includes('Error') ? 'var(--error)' : 'var(--success)' }}>
              {trainingStatus}
            </div>
          )}
        </div>
      </div>

      <main style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) 2fr', gap: '2rem' }}>
        {/* Input Panel */}
        <section className="glass-card h-fit">
          <h2 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Search size={22} color="var(--primary)" /> Scenario Input
          </h2>

          {error && (
            <div className="status-box" style={{ borderColor: 'var(--error)' }}>
               <span style={{ color: 'var(--error)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                 <AlertCircle size={18} /> {error}
               </span>
            </div>
          )}

          <form onSubmit={handlePredictAndOptimize}>
            <div className="form-group">
              <label>Select Target Areas (Hold Ctrl/Cmd to multi-select)</label>
              <select multiple value={selectedAreas} onChange={handleSelectArea} required>
                {areas.map(a => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </div>

            <div className="form-group" style={{ marginBottom: '2rem' }}>
              <label>Maximum Hospital Capacity (Patients)</label>
              <input 
                type="number" 
                value={capacity} 
                onChange={(e) => setCapacity(e.target.value)} 
                min="1" 
                required 
              />
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
               {loading ? <span className="loader"></span> : "Predict & Optimize Placement"}
            </button>
          </form>
        </section>

        {/* Results Panel */}
        <section>
          {loading && !results && (
            <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px' }}>
              <div className="loader" style={{ width: '40px', height: '40px', borderWidth: '4px', marginBottom: '1rem' }}></div>
              <p style={{ color: 'var(--text-muted)' }}>Running XGBoost inferences and solving MILP...</p>
            </div>
          )}

          {results && (
            <div className="glass-card" style={{ animation: 'fadeInUp 0.6s ease-out' }}>
              <div className="results-header">
                <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', justifyContent: 'center', marginBottom: '0.5rem' }}>
                  <Stethoscope size={28} color="var(--secondary)" /> Optimal Placement Strategy
                </h2>
                <div style={{ color: 'var(--success)', fontSize: '1.2rem', fontWeight: 'bold' }}>
                  {results.optimization.total_hospitals} Hospital(s) are enough to cover the {selectedAreas.length} selected areas.
                </div>
              </div>

              {/* High-level stats */}
              <div className="metric-cards">
                 <div className="metric-card glass-card" style={{ padding: '1rem' }}>
                   <h3>Hospitals Built</h3>
                   <div className="value">{results.optimization.total_hospitals}</div>
                 </div>
                 <div className="metric-card glass-card" style={{ padding: '1rem' }}>
                   <h3>Total Demand</h3>
                   <div className="value" style={{ color: '#c084fc' }}>
                     {Object.values(results.demands).reduce((a, b) => a + b.demand, 0)}
                   </div>
                 </div>
              </div>

              {/* Hub Locations */}
              <h3 style={{ marginBottom: '1rem', color: '#cbd5e1' }}>Designated Hospital Locations:</h3>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '2rem' }}>
                {results.optimization.hospitals.map(hub => (
                   <div key={hub} style={{ background: 'var(--primary)', padding: '0.5rem 1rem', borderRadius: '20px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem', boxShadow: '0 0 15px rgba(79, 70, 229, 0.4)' }}>
                     <Stethoscope size={16} /> {hub}
                   </div>
                ))}
              </div>

              {/* Area mapping */}
              <h3 style={{ marginBottom: '1rem', color: '#cbd5e1', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Network size={20} /> Area Coverage Mapping
              </h3>
              
              <div className="mapping-section">
                {results.optimization.assignments.map((assignment, i) => {
                   const demand = results.demands[assignment.area].demand;
                   const isHub = assignment.area === assignment.assigned_hospital;
                   
                   return (
                     <div className="mapping-card" key={i}>
                       <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                         <div>
                           <span style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{assignment.area}</span>
                           <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Demand: {demand}</div>
                         </div>
                       </div>
                       
                       <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                         <span style={{ color: 'var(--text-muted)' }}>→</span>
                         <span className="badge" style={{ backgroundColor: isHub ? 'rgba(16, 185, 129, 0.2)' : undefined, color: isHub ? 'var(--success)' : undefined }}>
                           {isHub ? 'Hosts Hospital' : `Assigned to ${assignment.assigned_hospital}`}
                         </span>
                       </div>
                     </div>
                   );
                })}
              </div>

            </div>
          )}

          {!results && !loading && (
            <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', opacity: 0.5 }}>
              <Network size={48} style={{ marginBottom: '1rem' }} />
              <p>Select areas and input capacity to see the optimal placement.</p>
            </div>
          )}

        </section>
      </main>
    </div>
  );
}
