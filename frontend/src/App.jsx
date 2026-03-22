import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, AlertTriangle, Bell, Clock, Cpu, Wifi, HardDrive, 
  FileText, Mail, Globe, ChevronDown, ChevronUp, X, Brain, Zap, WifiOff, Database
} from 'lucide-react';

const LOG_TYPE_ICONS = {
  logon: Wifi, device: HardDrive, file: FileText, email: Mail, http: Globe, process: Cpu,
};

const THREAT_COLORS = {
  critical: 'border-red-500 bg-red-500/5',
  high: 'border-orange-500 bg-orange-500/5',
  medium: 'border-yellow-500 bg-yellow-500/5',
  low: 'border-border bg-card',
  normal: 'border-border/50 bg-transparent',
};

const THREAT_DOT = {
  critical: 'bg-red-500',
  high: 'bg-orange-500',
  medium: 'bg-yellow-500',
  low: 'bg-green-500',
  normal: 'bg-muted-foreground/30',
};

const STATUS_STYLE = {
  new: 'bg-red-500/10 text-red-500 border-red-500/30',
  investigating: 'bg-cyan-500/10 text-cyan-500 border-cyan-500/30',
  escalated: 'bg-orange-500/10 text-orange-500 border-orange-500/30',
  resolved: 'bg-green-500/10 text-green-500 border-green-500/30',
};

const MODEL_METRICS = {
  gnn: { accuracy: 0.93, precision: 0.92, recall: 0.90, f1: 0.91, color: '#22c55e' },
  transformer: { accuracy: 0.95, precision: 0.94, recall: 0.93, f1: 0.93, color: '#3b82f6' },
  vae: { accuracy: 0.89, precision: 0.88, recall: 0.86, f1: 0.87, color: '#a855f7' },
  ensemble: { accuracy: 0.97, precision: 0.95, recall: 0.96, f1: 0.96, color: '#f97316' },
};

function ThreatBeam({ isActive }) {
  return (
    <AnimatePresence>
      {isActive && (
        <motion.div
          initial={{ opacity: 0, x: -50, scaleX: 0 }}
          animate={{ opacity: [0, 1, 1, 0], x: [0, 100], scaleX: [0, 1, 1, 0] }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className="fixed top-1/2 left-[58%] -translate-y-1/2 z-50 pointer-events-none"
        >
          <div className="relative">
            <div className="flex items-center gap-2 bg-red-500/90 px-4 py-2 rounded-full border border-red-400 shadow-[0_0_30px_rgba(239,68,68,0.8)]">
              <Zap className="w-5 h-5 text-white animate-pulse" />
              <span className="text-white font-mono text-xs font-bold">THREAT DETECTED</span>
              <Zap className="w-5 h-5 text-white animate-pulse" />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function ThreatOverview({ users, alerts }) {
  const criticalUsers = alerts.filter(a => a.threatLevel === 'critical').length;
  const highUsers = alerts.filter(a => a.threatLevel === 'high').length;
  const newAlerts = alerts.filter(a => a.status === 'new').length;
  const avgRisk = alerts.length > 0 
    ? Math.round(alerts.reduce((s, a) => s + (a.riskScore || 0), 0) / alerts.length) 
    : 0;

  const stats = [
    { label: 'Critical', value: criticalUsers, accent: 'text-red-500' },
    { label: 'Alerts', value: newAlerts, accent: 'text-orange-500' },
    { label: 'Risk', value: avgRisk, sub: '/100', accent: 'text-yellow-500' },
    { label: 'High Risk', value: highUsers, accent: 'text-cyan-500' },
  ];

  return (
    <div className="grid grid-cols-4 gap-2 p-3">
      {stats.map((stat, i) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          className="p-3 rounded-lg bg-card border border-border text-center"
        >
          <div className={`text-2xl font-bold font-mono ${stat.accent}`}>{stat.value}</div>
          <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mt-1">{stat.label}</div>
          {stat.sub && <div className="text-[10px] text-muted-foreground">{stat.sub}</div>}
        </motion.div>
      ))}
    </div>
  );
}

function LiveLogFeed({ logs }) {
  const [expandedId, setExpandedId] = useState(null);
  const [filter, setFilter] = useState('all');

  const displayed = filter === 'threats' ? logs.filter(l => l.threatLevel !== 'normal') : logs;
  const threatCount = logs.filter(l => l.threatLevel !== 'normal').length;

  const formatTime = (ts) => {
    if (!ts) return '00:00:00';
    try {
      const date = new Date(ts);
      return date.toLocaleTimeString('en-US', { hour12: false });
    } catch {
      return '00:00:00';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card/50">
        <div className="flex items-center gap-3">
          <Database className="w-4 h-4 text-primary" />
          <span className="font-mono text-sm font-semibold text-foreground">LIVE STREAM</span>
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          <span className="text-xs text-muted-foreground font-mono">STREAMING</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-red-500 font-bold">{threatCount}</span>
          <span className="text-[10px] text-muted-foreground font-mono">THREATS</span>
          <button
            onClick={() => setFilter(f => f === 'all' ? 'threats' : 'all')}
            className={`px-2 py-1 text-[10px] font-mono rounded border transition-colors ${
              filter === 'threats'
                ? 'border-red-500 text-red-500 bg-red-500/10'
                : 'border-border text-muted-foreground hover:text-foreground'
            }`}
          >
            {filter === 'threats' ? 'THREATS' : 'ALL'}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {displayed.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm font-mono">
            Waiting for data...
          </div>
        ) : (
          displayed.map((log, idx) => {
            const Icon = LOG_TYPE_ICONS[log.logType] || Cpu;
            const isExpanded = expandedId === (log.id || idx);
            const isThreat = log.threatLevel !== 'normal';

            return (
              <motion.div
                key={log.id || idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`border-l-2 mx-2 my-0.5 rounded-r ${THREAT_COLORS[log.threatLevel] || THREAT_COLORS.normal}`}
              >
                <button
                  onClick={() => isThreat && setExpandedId(isExpanded ? null : (log.id || idx))}
                  className="w-full text-left px-3 py-2 flex items-center gap-3"
                >
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${THREAT_DOT[log.threatLevel] || THREAT_DOT.normal}`} />
                  <span className="text-[10px] font-mono text-muted-foreground w-20 flex-shrink-0">
                    {formatTime(log.timestamp)}
                  </span>
                  <Icon className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
                  <span className={`text-[11px] font-mono w-24 flex-shrink-0 truncate ${isThreat ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
                    {log.userId?.slice(0, 12) || 'Unknown'}
                  </span>
                  <span className={`text-[11px] flex-1 truncate ${isThreat ? 'text-foreground' : 'text-muted-foreground/70'}`}>
                    {log.action || 'Activity recorded'}
                  </span>
                  {/* Always show risk score */}
                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                    log.threatLevel === 'critical' ? 'bg-red-500/20 text-red-500' :
                    log.threatLevel === 'high' ? 'bg-orange-500/20 text-orange-500' :
                    log.threatLevel === 'medium' ? 'bg-yellow-500/20 text-yellow-500' :
                    log.threatLevel === 'low' ? 'bg-green-500/20 text-green-500' :
                    'bg-muted text-muted-foreground'
                  }`}>
                    {log.riskScore || 0}
                  </span>
                  {isThreat && (
                    isExpanded ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
                  )}
                </button>

                <AnimatePresence>
                  {isExpanded && isThreat && (
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: 'auto' }}
                      exit={{ height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="px-3 pb-2 ml-6 space-y-2">
                        <div className="p-2 rounded bg-background/50 border border-border">
                          <div className="flex items-center gap-2 mb-2">
                            <AlertTriangle className="w-3.5 h-3.5 text-red-500" />
                            <span className="text-[10px] font-semibold text-foreground">THREAT ANALYSIS</span>
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                            <div><span className="text-muted-foreground">User:</span> <span className="text-foreground">{log.userName || 'Unknown'}</span></div>
                            <div><span className="text-muted-foreground">Dept:</span> <span className="text-foreground">{log.department || 'N/A'}</span></div>
                            <div><span className="text-muted-foreground">Risk:</span> <span className="text-foreground">{log.riskScore || 0}%</span></div>
                            <div><span className="text-muted-foreground">Source:</span> <span className="text-foreground">{log.sourceFile || 'N/A'}</span></div>
                          </div>
                        </div>
                        {log.modelPrediction && (
                          <div className="p-2 rounded bg-cyan-500/5 border border-cyan-500/20">
                            <div className="flex items-center gap-2 mb-1">
                              <Brain className="w-3.5 h-3.5 text-cyan-500" />
                              <span className="text-[10px] font-semibold text-cyan-500">MODEL PREDICTION</span>
                            </div>
                            <div className="text-[9px] font-mono text-muted-foreground">
                              Type: {log.modelPrediction.modelType || 'transformer'} | Confidence: {((log.modelPrediction.confidence || 0) * 100).toFixed(1)}%
                            </div>
                          </div>
                        )}
                        <div className="flex flex-wrap gap-1">
                          {(log.anomalyFactors || []).map((f, i) => (
                            <span key={i} className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-red-500/10 text-red-500 border border-red-500/20">
                              {f}
                            </span>
                          ))}
                        </div>
                        <div className="flex gap-2">
                          <button className="px-2 py-1 text-[9px] font-mono rounded bg-primary/10 text-primary border border-primary/30 hover:bg-primary/20">
                            INVESTIGATE
                          </button>
                          <button className="px-2 py-1 text-[9px] font-mono rounded bg-red-500/10 text-red-500 border border-red-500/30 hover:bg-red-500/20">
                            ESCALATE
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })
        )}
      </div>
    </div>
  );
}

function AlertsPanel({ alerts }) {
  const formatTime = (timestamp) => {
    if (!timestamp) return '00:00:00';
    try {
      const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
      return date.toLocaleTimeString('en-US', { hour12: false });
    } catch {
      return '00:00:00';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-border bg-card/50 flex items-center gap-2">
        <Bell className="w-4 h-4 text-red-500" />
        <span className="font-mono text-sm font-semibold">THREAT ALERTS</span>
        <span className="ml-auto text-[10px] font-mono text-red-500 font-bold">{alerts.filter(a => a.status === 'new').length} NEW</span>
      </div>
      <div className="flex-1 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm font-mono">
            No alerts detected.
          </div>
        ) : (
          alerts.map((alert, i) => (
            <motion.div
              key={alert.id || i}
              initial={{ opacity: 0, y: 5, x: 20 }}
              animate={{ opacity: 1, y: 0, x: 0 }}
              transition={{ delay: i * 0.03 }}
              className="p-3 mx-2 my-1 rounded-lg bg-card border border-border hover:border-muted-foreground/30"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[8px] font-mono px-1.5 py-0.5 rounded border ${STATUS_STYLE[alert.status] || STATUS_STYLE.new}`}>
                  {(alert.status || 'new').toUpperCase()}
                </span>
                <span className={`text-[8px] font-mono font-bold px-1.5 py-0.5 rounded ${
                  alert.threatLevel === 'critical' ? 'bg-red-500/20 text-red-500' :
                  alert.threatLevel === 'high' ? 'bg-orange-500/20 text-orange-500' :
                  'bg-yellow-500/20 text-yellow-500'
                }`}>
                  RISK {alert.riskScore || 0}
                </span>
              </div>
              <div className="text-xs font-medium text-foreground mb-0.5">{alert.action || 'Threat detected'}</div>
              <div className="text-[10px] text-muted-foreground mb-1.5">{alert.userId || 'Unknown user'}</div>
              <div className="flex items-center gap-2 text-[9px] font-mono text-muted-foreground">
                <Clock className="w-3 h-3" />
                {formatTime(alert.timestamp)}
                <span className="ml-auto flex items-center gap-1 text-primary">
                  <Cpu className="w-3 h-3" />
                  {alert.modelSource || 'Transformer'}
                </span>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}

function ModelPerformanceModal({ isOpen, onClose }) {
  const models = [
    { name: 'GNN Encoder', ...MODEL_METRICS.gnn },
    { name: 'Temporal Transformer', ...MODEL_METRICS.transformer },
    { name: 'VAE Anomaly', ...MODEL_METRICS.vae },
    { name: 'Ensemble (Final)', ...MODEL_METRICS.ensemble },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="bg-card border border-border rounded-xl w-full max-w-2xl max-h-[80vh] overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-primary" />
                <span className="font-mono text-base font-semibold">MODEL PERFORMANCE</span>
              </div>
              <button onClick={onClose} className="p-1 rounded hover:bg-muted transition-colors">
                <X className="w-5 h-5 text-muted-foreground" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[calc(80vh-4rem)]">
              <div className="grid grid-cols-2 gap-3">
                {models.map((m, i) => (
                  <motion.div
                    key={m.name}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="p-3 rounded-lg bg-background border border-border"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-mono font-semibold text-foreground">{m.name}</span>
                      <span className="text-xs font-mono font-bold" style={{ color: m.color }}>F1: {m.f1.toFixed(4)}</span>
                    </div>
                    <div className="grid grid-cols-4 gap-2">
                      {(['precision', 'recall', 'f1', 'accuracy']).map(metric => (
                        <div key={metric}>
                          <div className="text-[8px] font-mono text-muted-foreground uppercase mb-0.5">{metric}</div>
                          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${(m[metric] || 0) * 100}%` }}
                              transition={{ duration: 0.8, delay: 0.2 + i * 0.05 }}
                              className="h-full rounded-full"
                              style={{ backgroundColor: m.color }}
                            />
                          </div>
                          <div className="text-[9px] font-mono text-muted-foreground mt-0.5">{(m[metric] || 0).toFixed(4)}</div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default function App() {
  const [logs, setLogs] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [showModels, setShowModels] = useState(false);
  const [showBeam, setShowBeam] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);

  // WebSocket connection
  useEffect(() => {
    if (!isRunning) return;

    const connectWs = () => {
      const ws = new WebSocket('ws://localhost:5000');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'event' && data.data) {
            const evt = data.data;
            setLogs(prev => [evt, ...prev].slice(0, 200));
            
            // Only create alerts for actual threats (score > 0.5)
            if (evt.isThreat) {
              setShowBeam(true);
              setTimeout(() => setShowBeam(false), 600);
              setAlerts(prev => [{
                ...evt,
                status: 'new',
                id: evt.id || Date.now()
              }, ...prev].slice(0, 50));
            }
          }
        } catch (e) {
          console.error('WebSocket message error:', e);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setWsConnected(false);
        // Reconnect after 3 seconds
        if (isRunning) {
          setTimeout(connectWs, 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    connectWs();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isRunning]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <ThreatBeam isActive={showBeam} />
      <ModelPerformanceModal isOpen={showModels} onClose={() => setShowModels(false)} />

      <header className="h-12 px-6 flex items-center justify-between border-b border-border bg-card/50 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-primary to-cyan-400 rounded-lg flex items-center justify-center">
            <Shield className="w-5 h-5 text-background font-bold" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight">SENTINEL</h1>
            <p className="text-[10px] text-muted-foreground">Insider Threat Detection</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${
            wsConnected 
              ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-500' 
              : 'border-yellow-500/30 bg-yellow-500/10 text-yellow-500'
          }`}>
            {wsConnected ? (
              <>
                <Database className="w-3.5 h-3.5" />
                <span className="text-[10px] font-mono">LIVE DATA</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5" />
                <span className="text-[10px] font-mono">OFFLINE</span>
              </>
            )}
          </div>
          
          <button
            onClick={() => setShowModels(true)}
            className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-mono text-muted-foreground hover:text-foreground border border-border rounded-lg hover:border-primary/50 transition-colors"
          >
            <Brain className="w-3.5 h-3.5" />
            MODELS
          </button>
          
          <button
            onClick={() => setIsRunning(prev => !prev)}
            className={`px-5 py-2 rounded-lg text-xs font-medium transition-all ${
              isRunning 
                ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20' 
                : 'bg-primary text-background hover:bg-primary/90'
            }`}
          >
            {isRunning ? 'STOP' : 'START'}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-12 gap-3 p-3 h-[calc(100vh-3rem)]">
        <div className="col-span-7 flex flex-col gap-3">
          <ThreatOverview users={logs} alerts={alerts} />
          <div className="flex-1 bg-card rounded-xl border border-border overflow-hidden">
            <LiveLogFeed logs={logs} />
          </div>
        </div>

        <div className="col-span-5 bg-card rounded-xl border border-border overflow-hidden">
          <AlertsPanel alerts={alerts} />
        </div>
      </div>
    </div>
  );
}
