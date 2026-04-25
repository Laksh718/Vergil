/* ═══════════════════════════════════════════════════════════
   VERGIL — App Logic v4 (Theater Layout)
   ═══════════════════════════════════════════════════════════ */

const API = '';

// ── State ────────────────────────────────────────────────
let currentState = null;
let selectedNode = null;
let totalReward  = 0;
let autoTimer    = null;
let d3Sim        = null;
let episodeHistory = [];      // [{action,target,reward,step}]

// ── DOM shortcuts ────────────────────────────────────────
const $ = id => document.getElementById(id);

// ═══════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    loadScenarios();

    $('btn-reset').addEventListener('click', resetEpisode);
    $('ge-start-btn').addEventListener('click', resetEpisode);
    $('btn-auto').addEventListener('click', toggleAutoplay);
    $('btn-compare').addEventListener('click', openCompare);
    $('btn-close-compare').addEventListener('click', closeCompare);
    $('btn-run-compare').addEventListener('click', runComparison);
    $('btn-step-prev').addEventListener('click', () => compareStep(-1));
    $('btn-step-next').addEventListener('click', () => compareStep(+1));
    $('btn-cmp-auto').addEventListener('click', toggleCompareAuto);

    $('node-picker').addEventListener('change', e => {
        if (e.target.value) selectNode(e.target.value);
    });

    document.querySelectorAll('.ma-btn').forEach(btn => {
        btn.addEventListener('click', () => takeAction(btn.dataset.action));
    });
});

async function loadScenarios() {
    try {
        const data = await fetchJSON(`${API}/api/scenarios`);
        data.scenarios.forEach(s => {
            const o = document.createElement('option');
            o.value = s.scenario_id;
            o.textContent = `${s.scenario_id.replace('scenario_','').replace(/_/g,' ')}`;
            $('scenario-select').appendChild(o);
            const o2 = o.cloneNode(true);
            $('cmp-scenario-select').appendChild(o2);
        });
    } catch(e) { /* no scenarios endpoint — fine */ }
}

// ═══════════════════════════════════════════════════════════
//  RESET
// ═══════════════════════════════════════════════════════════
async function resetEpisode() {
    stopAutoplay();
    totalReward = 0;
    episodeHistory = [];
    selectedNode = null;

    const body = {};
    const sel = $('scenario-select').value;
    if (sel) body.scenario_id = sel;

    setLoading(true);
    try {
        const data = await fetchJSON(`${API}/api/reset`, { method: 'POST', body });
        currentState = data.state;

        clearFeed();
        clearTimeline();
        clearLog();
        feedSystem('Episode started — reviewing commitment requests in CDG.');
        renderScenarioHeader(currentState);
        renderAll(currentState, null);

        const pending = (currentState.graph?.nodes || []).filter(n => n.status === 'pending');
        if (pending.length) {
            selectNode(pending[0].id);
            pending.forEach(n => feedStakeholder(n));
        }
        $('graph-empty').classList.add('hidden');
    } catch(e) {
        feedSystem(`Reset failed: ${e.message}`, true);
    } finally {
        setLoading(false);
    }
}

// ═══════════════════════════════════════════════════════════
//  TAKE ACTION (manual)
// ═══════════════════════════════════════════════════════════
async function takeAction(actionType) {
    if (!currentState) { feedSystem('No episode loaded — click New Episode.', true); return; }
    if (actionType !== 'do_nothing' && !selectedNode) {
        feedSystem('Select a commitment node first.', true); return;
    }

    setActionsEnabled(false);
    try {
        const body = { action_type: actionType, target_node_id: selectedNode };
        if (actionType === 'counter_propose' && selectedNode) {
            const node = (currentState.graph?.nodes || []).find(n => n.id === selectedNode);
            if (node?.deadline) {
                const dl = new Date(node.deadline);
                dl.setHours(dl.getHours() + Math.round((node.estimated_duration_hours || 2) * 1.5));
                body.proposed_deadline = dl.toISOString();
            }
        }
        const data = await fetchJSON(`${API}/api/step`, { method: 'POST', body });
        handleStepResponse(data, actionType, null);
    } catch(e) {
        feedSystem(`Action failed: ${e.message}`, true);
    } finally {
        setActionsEnabled(true);
    }
}

// ═══════════════════════════════════════════════════════════
//  AGENT AUTO-STEP (uses /api/agent-step — LLM or heuristic)
// ═══════════════════════════════════════════════════════════
async function agentStep() {
    if (!currentState) return;
    try {
        const data = await fetchJSON(`${API}/api/agent-step`, { method: 'POST', body: {} });
        handleStepResponse(data, data.action, data.reasoning);
    } catch(e) {
        feedSystem(`Agent step failed: ${e.message}`, true);
        stopAutoplay();
    }
}

function handleStepResponse(data, actionType, reasoning) {
    if (data.detail) { feedSystem(`Error: ${data.detail}`, true); return; }

    currentState = data.state;
    const reward = data.reward || 0;
    totalReward += reward;

    const targetId = data.target_node_id || data.target;
    const nodes    = currentState.graph?.nodes || [];
    const node     = nodes.find(n => n.id === targetId);

    // Show agent reasoning block if available
    if (reasoning) feedThink(reasoning);

    // Show decision card
    feedDecision(actionType, node, reward, data.info?.stakeholder_responses);

    // Timeline entry
    pushTimeline(actionType, node?.label || targetId || '—', reward);

    // Log brief summary
    logAdd('agent', `${actionIcon(actionType)} ${node?.label || actionType}  (${reward >= 0 ? '+' : ''}${reward.toFixed(3)})`);

    // Cascade events
    const cascades = data.info?.cascade_events || [];
    if (cascades.length) {
        feedCascade(cascades);
        logAdd('danger', `⚠ Cascade: ${cascades.length} node(s) affected`);
    }

    // New pending from stakeholder responses
    const newPending = nodes.filter(n =>
        n.status === 'pending' &&
        !episodeHistory.some(h => h.nodeId === n.id)
    );
    newPending.forEach(n => feedStakeholder(n));

    episodeHistory.push({ step: currentState.step_number, actionType, nodeId: targetId, reward });

    renderAll(currentState, data);

    // Auto-select next pending
    const pending = nodes.filter(n => n.status === 'pending');
    if (pending.length && !pending.find(n => n.id === selectedNode)) selectNode(pending[0].id);

    if (data.terminated || data.truncated) {
        stopAutoplay();
        feedSystem('🏁 Episode complete! ' + episodeSummary());
        logAdd('success', '🏁 Episode complete — ' + episodeSummary());
    }
}

function episodeSummary() {
    if (!currentState) return '';
    const nodes = currentState.graph?.nodes || [];
    const done  = nodes.filter(n => n.status === 'completed').length;
    const total = nodes.filter(n => ['completed','failed','accepted'].includes(n.status)).length;
    return `${done}/${total} completed, total reward ${totalReward >= 0 ? '+' : ''}${totalReward.toFixed(2)}`;
}

// ═══════════════════════════════════════════════════════════
//  AUTOPLAY
// ═══════════════════════════════════════════════════════════
function toggleAutoplay() {
    if (autoTimer) {
        stopAutoplay();
    } else {
        $('btn-auto').textContent = '⏹ Stop Agent';
        $('btn-auto').classList.add('playing');
        autoTimer = setInterval(() => {
            if (!currentState) { stopAutoplay(); return; }
            agentStep();
        }, 1400);
        agentStep();
    }
}
function stopAutoplay() {
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    $('btn-auto').textContent = '▶ Auto-Play Agent';
    $('btn-auto').classList.remove('playing');
}

// ═══════════════════════════════════════════════════════════
//  RENDER ALL
// ═══════════════════════════════════════════════════════════
function renderAll(state, stepData) {
    renderTopbar(state);
    renderGraph(state);
    renderNodePicker(state);
    renderTrust(state);
    renderCapacity(state);
    renderReward(stepData);
    renderTargetDetail(state);
    renderGraphIndicators(state);
}

function renderTopbar(state) {
    $('stat-step').textContent = state.step_number || 0;

    const r = totalReward;
    const rEl = $('stat-reward');
    rEl.textContent = (r >= 0 ? '+' : '') + r.toFixed(2);
    rEl.style.color = r >= 0 ? 'var(--green)' : 'var(--red)';

    const sat = state.satisfiability_score;
    const satEl = $('stat-sat');
    if (sat != null) {
        const pct = Math.round(sat * 100);
        satEl.textContent = pct + '%';
        satEl.style.color = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--yellow)' : 'var(--red)';
    } else {
        satEl.textContent = '—'; satEl.style.color = '';
    }

    const load = state.cognitive_load;
    const ldEl = $('stat-load');
    if (load != null) {
        const pct = Math.round(load * 100);
        ldEl.textContent = pct + '%';
        ldEl.style.color = pct > 80 ? 'var(--red)' : pct > 50 ? 'var(--yellow)' : 'var(--green)';
    }

    $('badge-stage').textContent = `Stage ${state.curriculum_stage || 1}`;
}

function renderScenarioHeader(state) {
    const nodes = state.graph?.nodes || [];
    const n = nodes.length;
    const stakes = new Set(nodes.map(nd => nd.stakeholder_id).filter(Boolean));
    $('sh-title').textContent = `${n} commitment${n !== 1 ? 's' : ''} — ${stakes.size} stakeholder${stakes.size !== 1 ? 's' : ''}`;
    $('sh-sub').textContent   = `${state.available_hours_next_48h?.toFixed(1) || '—'}h available in 48h window`;
    $('sh-icon').textContent  = n > 3 ? '🌪' : n > 1 ? '⚡' : '💡';
}

function renderGraphIndicators(state) {
    const nodes = state.graph?.nodes || [];
    const pending   = nodes.filter(n => n.status === 'pending').length;
    const active    = nodes.filter(n => n.status === 'accepted').length;
    const failed    = nodes.filter(n => n.status === 'failed').length;

    const pEl = $('ghb-pending');
    const aEl = $('ghb-active');
    const fEl = $('ghb-failed');

    pEl.textContent = `${pending} pending`;
    pEl.style.color = pending > 0 ? 'var(--yellow)' : 'var(--text-3)';

    aEl.textContent = `${active} active`;
    aEl.style.color = active > 0 ? 'var(--blue)' : 'var(--text-3)';

    fEl.textContent = `${failed} failed`;
    fEl.style.color = failed > 0 ? 'var(--red)' : 'var(--text-3)';
}

// ═══════════════════════════════════════════════════════════
//  D3 GRAPH
// ═══════════════════════════════════════════════════════════
function renderGraph(state) {
    const graphData = state.graph;
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) return;

    const container = document.getElementById('graph-area');
    const W = container.clientWidth  || 600;
    const H = container.clientHeight || 400;

    const svg = d3.select('#graph-svg');
    svg.selectAll('*').remove();

    // Build maps for current positions (preserve layout on re-render)
    const prevPos = {};
    if (d3Sim) {
        d3Sim.stop();
        d3Sim.nodes().forEach(n => { prevPos[n.id] = { x: n.x, y: n.y }; });
    }

    const defs = svg.append('defs');
    // Arrow marker
    defs.append('marker')
        .attr('id', 'arrow')
        .attr('viewBox', '0 -4 8 8').attr('refX', 22).attr('refY', 0)
        .attr('markerWidth', 5).attr('markerHeight', 5).attr('orient', 'auto')
        .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', '#5b6b82');

    defs.append('marker')
        .attr('id', 'arrow-red')
        .attr('viewBox', '0 -4 8 8').attr('refX', 22).attr('refY', 0)
        .attr('markerWidth', 5).attr('markerHeight', 5).attr('orient', 'auto')
        .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', 'var(--red)');

    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom()
        .scaleExtent([0.4, 3])
        .on('zoom', e => g.attr('transform', e.transform))
    );

    const nodes = graphData.nodes.map(n => ({
        ...n,
        x: prevPos[n.id]?.x || W/2 + (Math.random()-0.5)*200,
        y: prevPos[n.id]?.y || H/2 + (Math.random()-0.5)*200,
    }));
    const links = (graphData.edges || []).map(e => ({...e}));

    // Links
    const link = g.append('g').attr('class', 'links')
        .selectAll('line').data(links).join('line')
        .attr('class', d => `link ${d.edge_type || 'dependency'}`)
        .attr('marker-end', d => d.edge_type === 'conflict' ? 'url(#arrow-red)' : 'url(#arrow)');

    // Node groups
    const node = g.append('g').attr('class', 'nodes')
        .selectAll('g').data(nodes).join('g')
        .attr('class', d => `node status-${d.status}${d.id === selectedNode ? ' selected' : ''}`)
        .call(d3.drag()
            .on('start', (e,d) => { if (!e.active) d3Sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
            .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
            .on('end',   (e,d) => { if (!e.active) d3Sim.alphaTarget(0); d.fx=null; d.fy=null; })
        )
        .on('click', (e, d) => { e.stopPropagation(); selectNode(d.id); });

    const radius = d => 14 + (d.urgency || 0.5) * 8;

    node.append('circle').attr('r', radius);

    // Urgency ring
    node.append('circle')
        .attr('class', 'urgency-ring')
        .attr('r', d => radius(d) + 5)
        .attr('stroke', d => {
            const u = d.urgency || 0;
            return u > 0.7 ? 'var(--red)' : u > 0.4 ? 'var(--yellow)' : 'var(--green)';
        })
        .attr('stroke-opacity', d => (d.urgency || 0) * 0.6)
        .attr('fill', 'none')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '3,3');

    // Labels
    node.append('text')
        .attr('dy', '-1px')
        .text(d => d.label?.length > 12 ? d.label.slice(0, 10) + '…' : (d.label || d.id));

    node.append('text')
        .attr('class', 'node-sublabel')
        .attr('dy', '14px')
        .text(d => {
            const hrs = d.estimated_duration_hours;
            return hrs ? `${hrs}h` : '';
        });

    // Force simulation
    d3Sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(0.5))
        .force('charge', d3.forceManyBody().strength(-280))
        .force('center', d3.forceCenter(W/2, H/2))
        .force('collide', d3.forceCollide(d => radius(d) + 18))
        .on('tick', () => {
            link
                .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });
}

// ═══════════════════════════════════════════════════════════
//  NODE PICKER + SELECTION
// ═══════════════════════════════════════════════════════════
function renderNodePicker(state) {
    const picker = $('node-picker');
    const prev   = picker.value;
    picker.innerHTML = '<option value="">— select commitment —</option>';

    (state.graph?.nodes || []).forEach(n => {
        const o   = document.createElement('option');
        o.value   = n.id;
        const dur = n.estimated_duration_hours ? `${n.estimated_duration_hours}h` : '';
        o.textContent = `[${n.status}] ${n.label || n.id} ${dur}`;
        if (n.status !== 'pending') o.style.color = '#5b6b82';
        picker.appendChild(o);
    });
    if (prev) picker.value = prev;
}

function selectNode(nodeId) {
    selectedNode = nodeId;
    $('node-picker').value = nodeId;

    // Highlight in graph
    d3.selectAll('.node')
        .classed('selected', d => d.id === nodeId);

    renderTargetDetail(currentState);
}

function renderTargetDetail(state) {
    const el = $('target-detail');
    if (!selectedNode || !state) { el.innerHTML = '<div class="td-empty">Click a graph node or select from dropdown</div>'; return; }

    const node = (state.graph?.nodes || []).find(n => n.id === selectedNode);
    if (!node) { el.innerHTML = '<div class="td-empty">Node not found</div>'; return; }

    const dl = node.deadline ? new Date(node.deadline).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : 'none';
    const urgPct = Math.round((node.urgency || 0) * 100);

    el.innerHTML = `
        <div class="td-name">${node.label || node.id}</div>
        <div class="td-row"><span class="td-k">Status</span><span class="td-v"><span class="td-status ${node.status}">${node.status}</span></span></div>
        <div class="td-row"><span class="td-k">Duration</span><span class="td-v">${node.estimated_duration_hours || '?'}h</span></div>
        <div class="td-row"><span class="td-k">Deadline</span><span class="td-v">${dl}</span></div>
        <div class="td-row"><span class="td-k">Urgency</span><span class="td-v" style="color:${urgPct>70?'var(--red)':urgPct>40?'var(--yellow)':'var(--green)'}">${urgPct}%</span></div>
        <div class="td-row"><span class="td-k">Stakeholder</span><span class="td-v">${node.stakeholder_id || '—'}</span></div>
        ${node.type ? `<div class="td-row"><span class="td-k">Type</span><span class="td-v">${node.type}</span></div>` : ''}
    `;
}

// ═══════════════════════════════════════════════════════════
//  TRUST BARS
// ═══════════════════════════════════════════════════════════
function renderTrust(state) {
    const entries = state.trust_entries || {};
    const list    = $('trust-list');
    list.innerHTML = '';

    const scores = Object.values(entries).map(e => e.trust_score || 0);
    const avg    = scores.length ? (scores.reduce((a,b)=>a+b,0)/scores.length) : null;
    const avgBadge = $('trust-avg-badge');
    if (avg !== null) {
        avgBadge.textContent = `avg ${(avg*100).toFixed(0)}%`;
        avgBadge.style.background = avg >= 0.6 ? 'hsla(142,50%,20%,0.3)' : avg >= 0.4 ? 'hsla(38,60%,20%,0.3)' : 'hsla(0,50%,20%,0.3)';
        avgBadge.style.color = avg >= 0.6 ? 'var(--green)' : avg >= 0.4 ? 'var(--yellow)' : 'var(--red)';
    }

    Object.entries(entries).forEach(([sid, te]) => {
        const score = te.trust_score || 0;
        const pct   = Math.round(score * 100);
        const cls   = score >= 0.65 ? 'high' : score >= 0.45 ? 'medium' : score >= 0.25 ? 'low' : 'critical';

        const md = state.multidim_trust?.[sid];
        let dimsHtml = '';
        if (md) {
            dimsHtml = `
                <div class="te-dims">
                    <span class="te-dim">R:<span>${(md.reliability*100).toFixed(0)}</span></span>
                    <span class="te-dim">C:<span>${(md.competence*100).toFixed(0)}</span></span>
                    <span class="te-dim">B:<span>${(md.benevolence*100).toFixed(0)}</span></span>
                </div>`;
        }

        list.insertAdjacentHTML('beforeend', `
            <div class="trust-entry">
                <div class="te-header">
                    <span class="te-name">${sid}</span>
                    <span class="te-score ${cls}">${pct}%</span>
                </div>
                <div class="te-bar-track">
                    <div class="te-bar-fill ${cls}" style="width:${pct}%"></div>
                </div>
                ${dimsHtml}
            </div>
        `);
    });

    if (!Object.keys(entries).length) {
        list.innerHTML = '<div style="color:var(--text-3);font-size:11px;padding:4px 0">No stakeholders yet</div>';
    }
}

// ═══════════════════════════════════════════════════════════
//  CAPACITY GAUGE
// ═══════════════════════════════════════════════════════════
function renderCapacity(state) {
    const avail     = state.available_hours_next_48h || 8;
    const nodes     = state.graph?.nodes || [];
    const committed = nodes
        .filter(n => ['accepted','in_progress'].includes(n.status))
        .reduce((s, n) => s + (n.estimated_duration_hours || 0), 0);

    const pct  = Math.min(100, Math.round((committed / avail) * 100));
    const cls  = pct >= 90 ? 'crit' : pct >= 70 ? 'warn' : '';

    $('cap-committed').textContent = committed.toFixed(1) + 'h';
    $('cap-available').textContent = avail.toFixed(1) + 'h';

    const fill = $('cap-bar-fill');
    fill.style.width = pct + '%';
    fill.className   = 'cap-bar-fill' + (cls ? ' ' + cls : '');
}

// ═══════════════════════════════════════════════════════════
//  REWARD BREAKDOWN
// ═══════════════════════════════════════════════════════════
function renderReward(stepData) {
    const el = $('reward-display');
    if (!stepData?.reward_components && !stepData?.info?.reward_components) {
        return;
    }

    const rc = stepData.info?.reward_components || stepData.reward_components;
    const r  = stepData.reward || 0;

    const rClass = r >= 0 ? 'pos' : 'neg';
    const rSign  = r >= 0 ? '+' : '';

    const rows = [
        { k: 'Fulfillment',     v: rc?.fulfillment     || 0 },
        { k: 'Trust Δ',         v: rc?.trust_delta     || 0 },
        { k: 'Proactive',       v: rc?.proactive       || 0 },
        { k: 'Accuracy',        v: rc?.feasibility_acc || 0 },
        { k: '— Broken',        v: -(rc?.broken_penalty || 0) },
        { k: '— Over-refusal',  v: -(rc?.overrefusal_penalty || 0) },
        { k: '— Silent drop',   v: -(rc?.silent_drop_penalty || 0) },
    ];

    el.innerHTML = `
        <div class="rwd-total ${rClass}">${rSign}${r.toFixed(4)}</div>
        ${rows.map(row => {
            const vCls = row.v > 0.001 ? 'pos' : row.v < -0.001 ? 'neg' : 'zero';
            const vSign = row.v >= 0 ? '+' : '';
            return `<div class="rwd-row">
                <span class="rwd-key">${row.k}</span>
                <span class="rwd-val ${vCls}">${vSign}${row.v.toFixed(4)}</span>
            </div>`;
        }).join('')}
    `;
}

// ═══════════════════════════════════════════════════════════
//  CONVERSATION FEED
// ═══════════════════════════════════════════════════════════
function clearFeed() {
    $('message-feed').innerHTML = '';
    $('feed-empty').classList.remove('hidden');
}

function feedMsg(html) {
    const feed = $('message-feed');
    $('feed-empty').classList.add('hidden');
    feed.insertAdjacentHTML('beforeend', html);
    feed.scrollTop = feed.scrollHeight;
}

function feedSystem(text, isError = false) {
    feedMsg(`<div class="msg msg-system${isError ? ' msg-alert' : ''}">${text}</div>`);
    logAdd(isError ? 'danger' : 'system', text);
}

function feedStakeholder(node) {
    const urgPct = Math.round((node.urgency || 0) * 100);
    const dl     = node.deadline
        ? new Date(node.deadline).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})
        : 'flexible deadline';
    feedMsg(`
        <div class="msg msg-stakeholder">
            <div class="msg-from">${node.stakeholder_id || 'Stakeholder'}</div>
            <div class="msg-body">Can you handle "<strong>${node.label || node.id}</strong>"?</div>
            <div class="msg-meta">${node.estimated_duration_hours || '?'}h · due ${dl} · urgency ${urgPct}%</div>
        </div>
    `);
}

function feedThink(reasoning) {
    // Parse structured reasoning into steps if it contains numbered lines
    const lines = reasoning.split('\n').filter(l => l.trim());
    const stepsHtml = lines.map(l => `<div class="think-step">${l.trim()}</div>`).join('');
    feedMsg(`
        <div class="msg msg-think">
            <div class="think-header">🧠 Agent Reasoning</div>
            <div class="think-body">${stepsHtml || reasoning}</div>
        </div>
    `);
}

function feedDecision(actionType, node, reward, stakeholderResponses) {
    const icons = { accept:'✅', decline:'❌', counter_propose:'🔄', do_nothing:'⏳', renegotiate:'🤝' };
    const labels= { accept:'Accepted', decline:'Declined', counter_propose:'Counter-proposed', do_nothing:'Waited', renegotiate:'Renegotiated' };
    const isPos = actionType === 'accept' || actionType === 'counter_propose';
    const rSign = reward >= 0 ? '+' : '';

    let responsesHtml = '';
    if (stakeholderResponses) {
        Object.entries(stakeholderResponses).forEach(([sid, msg]) => {
            if (msg) responsesHtml += `<div style="margin-top:4px;font-size:11px;color:var(--text-3)"><em>${sid}: "${msg}"</em></div>`;
        });
    }

    feedMsg(`
        <div class="msg msg-decision ${isPos ? '' : 'negative'}">
            <div class="md-action">${icons[actionType] || '•'} ${labels[actionType] || actionType}</div>
            <div class="md-target">${node ? `"${node.label || node.id}"` : '—'}</div>
            ${responsesHtml}
            <div class="md-reward">reward ${rSign}${reward.toFixed(4)} · total ${totalReward >= 0 ? '+' : ''}${totalReward.toFixed(2)}</div>
        </div>
    `);
}

function feedCascade(events) {
    const n = events.filter(e => e.cascaded).length;
    if (!n) return;
    feedMsg(`<div class="msg msg-cascade">⚠ Cascade failure propagated to ${n} node${n>1?'s':''}!</div>`);
}

// ═══════════════════════════════════════════════════════════
//  DECISION TIMELINE
// ═══════════════════════════════════════════════════════════
function clearTimeline() { $('timeline-track').innerHTML = ''; }

function pushTimeline(actionType, label, reward) {
    const track = $('timeline-track');
    const icons = { accept:'✅', decline:'❌', counter_propose:'🔄', do_nothing:'⏳' };
    const step  = episodeHistory.length + 1;
    const rCls  = reward >= 0 ? 'pos' : 'neg';
    const rSign = reward >= 0 ? '+' : '';

    if (track.children.length > 0) {
        track.insertAdjacentHTML('beforeend', '<div class="tl-connector"></div>');
    }

    track.insertAdjacentHTML('beforeend', `
        <div class="tl-step ${actionType}" title="Step ${step}: ${actionType} — ${label}">
            <div class="tl-icon">${icons[actionType] || '•'}</div>
            <div class="tl-label2">s${step}</div>
            <div class="tl-reward ${rCls}">${rSign}${reward.toFixed(2)}</div>
        </div>
    `);

    track.scrollLeft = track.scrollWidth;
}

// ═══════════════════════════════════════════════════════════
//  EVENT LOG (right panel)
// ═══════════════════════════════════════════════════════════
function clearLog() { $('log-list').innerHTML = ''; }

function logAdd(type, text) {
    const el = document.createElement('div');
    el.className = `log-item ${type}`;
    el.textContent = text;
    const list = $('log-list');
    list.appendChild(el);
    while (list.children.length > 60) list.removeChild(list.firstChild);
    list.scrollTop = list.scrollHeight;
}

// ═══════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════
async function fetchJSON(url, { method = 'GET', body } = {}) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res  = await fetch(url, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function setLoading(on) {
    $('btn-reset').textContent = on ? '⏳ Loading…' : 'New Episode';
    $('btn-reset').disabled    = on;
}

function setActionsEnabled(on) {
    document.querySelectorAll('.ma-btn').forEach(b => b.disabled = !on);
}

function actionIcon(type) {
    return { accept:'✅', decline:'❌', counter_propose:'🔄', do_nothing:'⏳' }[type] || '•';
}

// ═══════════════════════════════════════════════════════════
//  COMPARE MODE
// ═══════════════════════════════════════════════════════════
let compareData     = null;
let compareStepIdx  = 0;
let compareAutoTimer = null;

const SCENARIO_DESCS = {
    scenario_04_deadline_crunch:             { icon:'⏰', name:'Deadline Crunch',          desc:'Back-to-back deadlines — agent must triage' },
    scenario_07_simultaneous_infeasibility:  { icon:'💥', name:'Simultaneous Infeasibility',desc:'3 requests arrive at once — together impossible' },
    scenario_10_deadline_cascade:            { icon:'🌊', name:'Deadline Cascade Chain',    desc:'A→B→C dependency chain — one slip cascades' },
    scenario_11_impossible_math:             { icon:'🧮', name:'Impossible Math',           desc:'11.5h of work in 6h window — must decline' },
    scenario_12_force_majeure_recovery:      { icon:'🚨', name:'Force Majeure Recovery',    desc:'P0 incident blocks 7h mid-episode — renegotiate everything' },
};

function openCompare() {
    $('compare-overlay').classList.remove('hidden');
    updateCmpScenarioMeta();
}

function closeCompare() {
    stopCompareAuto();
    $('compare-overlay').classList.add('hidden');
    compareData = null;
}

$('cmp-scenario-select')?.addEventListener('change', updateCmpScenarioMeta);

function updateCmpScenarioMeta() {
    const id   = $('cmp-scenario-select').value;
    const meta = SCENARIO_DESCS[id] || { icon:'⚡', name: id.replace('scenario_','').replace(/_/g,' '), desc:'' };
    $('cmp-scenario-icon').textContent = meta.icon;
    $('cmp-scenario-name').textContent = meta.name;
    $('cmp-scenario-desc').textContent = meta.desc;
}

async function runComparison() {
    stopCompareAuto();
    const scenarioId = $('cmp-scenario-select').value;
    $('cmp-loading').classList.remove('hidden');
    $('cmp-body').classList.add('hidden');

    try {
        const data = await fetchJSON(`${API}/api/compare`, {
            method: 'POST',
            body: { scenario_id: scenarioId },
        });
        compareData    = data;
        compareStepIdx = 0;

        $('cmp-loading').classList.add('hidden');
        $('cmp-body').classList.remove('hidden');

        renderCmpDeltas(data);
        renderCmpStep(0);
        $('cmp-step-label').textContent = `Step 1 / ${Math.max(data.naive.steps.length, data.vergil.steps.length)}`;
    } catch(e) {
        $('cmp-loading').innerHTML = `<p style="color:var(--red)">Error: ${e.message}</p>`;
    }
}

function renderCmpDeltas(data) {
    const n = data.naive.metrics;
    const v = data.vergil.metrics;

    const rDelta    = (v.total_reward    || 0) - (n.total_reward    || 0);
    const satDelta  = (v.final_sat       || 0) - (n.final_sat       || 0);
    const failAvoid = (n.n_failed        || 0) - (v.n_failed        || 0);
    const trustDelta= (v.avg_trust       || 0) - (n.avg_trust       || 0);

    function fmt(val, isCount = false) {
        const sign = val >= 0 ? '+' : '';
        return isCount ? `${val >= 0 ? '+' : ''}${val}` : `${sign}${val.toFixed(2)}`;
    }
    function cls(val) { return val > 0 ? 'better' : val < 0 ? 'worse' : ''; }

    $('dv-reward').textContent = fmt(rDelta);
    $('dv-reward').className   = `dr-val ${cls(rDelta)}`;

    $('dv-sat').textContent    = fmt(satDelta * 100) + '%';
    $('dv-sat').className      = `dr-val ${cls(satDelta)}`;

    $('dv-fail').textContent   = fmt(failAvoid, true);
    $('dv-fail').className     = `dr-val ${cls(failAvoid)}`;

    $('dv-trust').textContent  = fmt(trustDelta * 100) + '%';
    $('dv-trust').className    = `dr-val ${cls(trustDelta)}`;

    // Verdict
    const improved = [rDelta > 0, satDelta > 0, failAvoid >= 0, trustDelta > 0].filter(Boolean).length;
    $('cmp-verdict').textContent =
        improved >= 3 ? '✅ VERGIL significantly outperforms naive agent' :
        improved >= 2 ? '↑ VERGIL shows clear improvement' :
        '~ Results comparable — try a harder scenario';

    // Naive & VERGIL final stats
    renderSideStats('naive-stats',  n);
    renderSideStats('vergil-stats', v);

    // Draw final CDG states
    renderMiniGraph('#cmp-svg-naive',  data.naive.final_graph,  'naive');
    renderMiniGraph('#cmp-svg-vergil', data.vergil.final_graph, 'vergil');
}

function renderSideStats(elId, metrics) {
    $(`${elId}`).innerHTML = `
        <div class="css-stat"><div class="css-label">Reward</div>
            <div class="css-val" style="color:${(metrics.total_reward||0)>=0?'var(--green)':'var(--red)'}">${(metrics.total_reward||0) >= 0 ? '+' : ''}${(metrics.total_reward||0).toFixed(2)}</div></div>
        <div class="css-stat"><div class="css-label">SAT</div>
            <div class="css-val">${Math.round((metrics.final_sat||0)*100)}%</div></div>
        <div class="css-stat"><div class="css-label">Failed</div>
            <div class="css-val" style="color:${(metrics.n_failed||0)>0?'var(--red)':'var(--green)'}">${metrics.n_failed||0}</div></div>
        <div class="css-stat"><div class="css-label">Trust</div>
            <div class="css-val">${Math.round((metrics.avg_trust||0)*100)}%</div></div>
    `;
}

function renderCmpStep(idx) {
    if (!compareData) return;
    const nSteps = compareData.naive.steps  || [];
    const vSteps = compareData.vergil.steps || [];
    const total  = Math.max(nSteps.length, vSteps.length);

    compareStepIdx = Math.max(0, Math.min(idx, total - 1));
    $('cmp-step-label').textContent = `Step ${compareStepIdx + 1} / ${total}`;

    const nStep = nSteps[compareStepIdx];
    const vStep = vSteps[compareStepIdx];

    function stepHtml(step, isVergil) {
        if (!step) return '<em style="color:var(--text-3)">No action</em>';
        const icon = actionIcon(step.action);
        const r    = step.reward || 0;
        const rS   = r >= 0 ? '+' : '';
        if (isVergil && step.reasoning) {
            return `${icon} <strong>${step.action}</strong> → ${step.target || '—'}<br>
                <span style="color:#c084fc;margin-top:3px;display:block">🧠 ${step.reasoning}</span>
                <span style="color:var(--text-3)">${rS}${r.toFixed(3)}</span>`;
        }
        return `${icon} <strong>${step.action}</strong> → ${step.target || '—'}<span style="color:var(--text-3);margin-left:8px">${rS}${r.toFixed(3)}</span>`;
    }

    $('naive-step-display').innerHTML   = stepHtml(nStep, false);
    $('vergil-step-display').innerHTML  = stepHtml(vStep, true);

    // If any naive step caused a failure, animate cascade
    if (nStep?.caused_failure) {
        $('cmp-svg-naive').classList.add('cascade-active');
        setTimeout(() => $('cmp-svg-naive').classList.remove('cascade-active'), 800);
    }
}

function compareStep(delta) {
    renderCmpStep(compareStepIdx + delta);
}

function toggleCompareAuto() {
    const btn = $('btn-cmp-auto');
    if (compareAutoTimer) {
        stopCompareAuto();
    } else {
        btn.textContent = '⏹ Stop';
        btn.classList.add('playing');
        compareAutoTimer = setInterval(() => {
            const total = Math.max(
                compareData?.naive.steps.length  || 0,
                compareData?.vergil.steps.length || 0
            );
            if (compareStepIdx >= total - 1) { stopCompareAuto(); return; }
            renderCmpStep(compareStepIdx + 1);
        }, 1500);
    }
}

function stopCompareAuto() {
    if (compareAutoTimer) { clearInterval(compareAutoTimer); compareAutoTimer = null; }
    const btn = $('btn-cmp-auto');
    if (btn) { btn.textContent = 'Auto ▶'; btn.classList.remove('playing'); }
}

function renderMiniGraph(svgSelector, graphData, side) {
    if (!graphData || !graphData.nodes?.length) return;

    const svgEl    = document.querySelector(svgSelector);
    if (!svgEl) return;
    const W = svgEl.clientWidth  || 500;
    const H = svgEl.clientHeight || 300;

    const svg = d3.select(svgSelector);
    svg.selectAll('*').remove();

    const g     = svg.append('g');
    const nodes = graphData.nodes.map(n => ({...n, x: W/2 + (Math.random()-.5)*200, y: H/2 + (Math.random()-.5)*200 }));
    const links = (graphData.edges || []).map(e => ({...e}));

    const colorByStatus = s => ({
        pending:   '#eab308', accepted: '#3b82f6',
        completed: '#22c55e', failed:   '#ef4444',
    }[s] || '#5b6b82');

    const link = g.append('g').selectAll('line').data(links).join('line')
        .attr('stroke', '#334155').attr('stroke-width', 1.5).attr('stroke-opacity', 0.5);

    const node = g.append('g').selectAll('g').data(nodes).join('g');

    node.append('circle')
        .attr('r', d => 10 + (d.urgency||0.5)*6)
        .attr('fill', d => `${colorByStatus(d.status)}22`)
        .attr('stroke', d => colorByStatus(d.status))
        .attr('stroke-width', d => d.status === 'failed' ? 3 : 1.5)
        .style('filter', d => d.status === 'failed' && side === 'naive'
            ? 'drop-shadow(0 0 8px rgba(239,68,68,0.8))' : 'none');

    node.append('text')
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
        .attr('fill', '#94a3b8').attr('font-size', '9px').attr('pointer-events', 'none')
        .text(d => d.label?.slice(0,8) || d.id?.slice(0,6));

    const sim = d3.forceSimulation(nodes)
        .force('link',    d3.forceLink(links).id(d => d.id).distance(80))
        .force('charge',  d3.forceManyBody().strength(-180))
        .force('center',  d3.forceCenter(W/2, H/2))
        .force('collide', d3.forceCollide(24))
        .on('tick', () => {
            link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
                .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
            node.attr('transform', d=>`translate(${d.x},${d.y})`);
        });

    // Stop after settling
    setTimeout(() => sim.stop(), 3000);
}
