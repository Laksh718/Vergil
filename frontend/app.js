/* ═══════════════════════════════════════════════════════════
   VERGIL — App Logic v3 (Complete Rebuild)
   ═══════════════════════════════════════════════════════════ */

const API = '';
let currentState = null;
let selectedNode = null;
let totalReward = 0;
let autoplayTimer = null;
let simulation = null;

// ── DOM Elements ────────────────────────────────────────────
const $ = id => document.getElementById(id);
const onboarding   = $('onboarding');
const app          = $('app');
const btnStart     = $('btn-start');
const btnReset     = $('btn-reset');
const btnAuto      = $('btn-auto');
const scenarioSel  = $('scenario-select');
const nodePicker   = $('node-picker');
const situationText= $('situation-text');
const situationIcon= $('situation-icon');
const statStep     = $('stat-step');
const statReward   = $('stat-reward');
const statSat      = $('stat-sat');
const badgeStage   = $('badge-stage');
const targetInfo   = $('target-info');
const trustList    = $('trust-list');
const rewardNumber = $('reward-number');
const rewardBreakdown = $('reward-breakdown');
const logList      = $('log-list');
const graphEmpty   = $('graph-empty');

// ═══════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    loadScenarios();
    btnStart.addEventListener('click', () => {
        onboarding.classList.add('hidden');
        app.classList.remove('hidden');
    });
    btnReset.addEventListener('click', resetEpisode);
    btnAuto.addEventListener('click', toggleAutoplay);
    nodePicker.addEventListener('change', (e) => {
        if (e.target.value) selectNode(e.target.value);
    });

    // Action buttons
    document.querySelectorAll('.act-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            if (action) takeAction(action);
        });
    });
});

async function loadScenarios() {
    try {
        const res = await fetch(`${API}/api/scenarios`);
        const data = await res.json();
        data.scenarios.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.scenario_id;
            opt.textContent = `${s.scenario_id} (${s.n_commitments} tasks, ${s.n_stakeholders} people)`;
            scenarioSel.appendChild(opt);
        });
    } catch(e) {
        console.warn('Could not load scenarios:', e);
    }
}

// ═══════════════════════════════════════════════════════════
//  RESET
// ═══════════════════════════════════════════════════════════

async function resetEpisode() {
    totalReward = 0;
    selectedNode = null;
    clearAutoplay();

    const body = {};
    if (scenarioSel.value) body.scenario_id = scenarioSel.value;

    try {
        btnReset.textContent = '⏳ Loading...';
        btnReset.disabled = true;

        const res = await fetch(`${API}/api/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        currentState = data.state;

        logClear();
        logAdd('system', 'New episode started. Review the commitment requests in the graph.');

        updateAll(currentState, null);

        // Auto-select first pending node
        const pending = currentState.graph?.nodes?.filter(n => n.status === 'pending') || [];
        if (pending.length > 0) {
            selectNode(pending[0].id);
        }

    } catch(e) {
        logAdd('danger', `Reset failed: ${e.message}`);
    } finally {
        btnReset.textContent = '🔄 New Episode';
        btnReset.disabled = false;
    }
}

// ═══════════════════════════════════════════════════════════
//  STEP (take action)
// ═══════════════════════════════════════════════════════════

async function takeAction(actionType) {
    if (!currentState) {
        logAdd('danger', 'No episode loaded. Click "New Episode" first.');
        return;
    }

    const targetId = selectedNode;

    // Validate action
    if (actionType !== 'do_nothing' && !targetId) {
        logAdd('danger', 'Select a commitment node first.');
        shake($('card-target'));
        return;
    }

    // Disable buttons during request
    setActionsEnabled(false);

    try {
        const body = {
            action_type: actionType,
            target_node_id: targetId,
        };

        // For counter_propose, always add a proposed deadline
        if (actionType === 'counter_propose' && targetId) {
            const node = currentState.graph?.nodes?.find(n => n.id === targetId);
            if (node) {
                let dl;
                if (node.deadline) {
                    dl = new Date(node.deadline);
                    dl.setHours(dl.getHours() + Math.round(node.estimated_duration_hours * 1.5));
                } else {
                    // No deadline — propose one based on duration
                    dl = new Date(currentState.current_time);
                    dl.setHours(dl.getHours() + Math.round(node.estimated_duration_hours * 2));
                }
                body.proposed_deadline = dl.toISOString();
            }
        }

        const res = await fetch(`${API}/api/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();

        if (data.detail) {
            logAdd('danger', `Error: ${data.detail}`);
            setActionsEnabled(true);
            return;
        }

        currentState = data.state;
        totalReward += data.reward;

        // Log the action
        const actionNames = {
            accept: '✅ Accepted', decline: '❌ Declined',
            counter_propose: '🔄 Counter-proposed', do_nothing: '⏳ Waited',
            renegotiate: '🤝 Renegotiated', clarify: '❓ Clarified',
        };
        const label = currentState.graph?.nodes?.find(n => n.id === targetId)?.label || targetId || '';
        logAdd('agent', `${actionNames[actionType] || actionType} "${label}"`);

        // Log stakeholder responses
        const responses = data.info?.stakeholder_responses || {};
        for (const [sid, msg] of Object.entries(responses)) {
            if (msg) logAdd('response', `${sid}: "${msg}"`);
        }

        // Log reward
        const rSign = data.reward >= 0 ? '+' : '';
        logAdd('system', `Reward: ${rSign}${data.reward.toFixed(3)} (total: ${rSign}${totalReward.toFixed(2)})`);

        // Check episodes over
        if (data.terminated || data.truncated) {
            logAdd('success', '🏁 Episode complete!');
            clearAutoplay();
            showEpisodeDone(data);
        }

        updateAll(currentState, data);

        // Auto-select next pending
        const pending = currentState.graph?.nodes?.filter(n => n.status === 'pending') || [];
        if (pending.length > 0 && !pending.find(n => n.id === selectedNode)) {
            selectNode(pending[0].id);
        }

    } catch(e) {
        logAdd('danger', `Action failed: ${e.message}`);
    } finally {
        setActionsEnabled(true);
    }
}

// ═══════════════════════════════════════════════════════════
//  UPDATE ALL UI
// ═══════════════════════════════════════════════════════════

function updateAll(state, stepData) {
    updateTopbar(state);
    updateSituation(state);
    updateGraph(state);
    updateNodePicker(state);
    updateTrust(state);
    updateReward(stepData);
    updateTargetInfo(state);
}

function updateTopbar(state) {
    statStep.textContent = state.step_number || 0;
    statReward.textContent = (totalReward >= 0 ? '+' : '') + totalReward.toFixed(2);
    statReward.style.color = totalReward >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';

    const sat = state.satisfiability_score;
    if (sat !== undefined && sat !== null) {
        const satPct = Math.round(sat * 100);
        statSat.textContent = satPct + '%';
        statSat.style.color = satPct >= 70 ? 'var(--accent-green)' : satPct >= 40 ? 'var(--accent-yellow)' : 'var(--accent-red)';
    } else {
        statSat.textContent = '—';
    }

    badgeStage.textContent = `Stage ${state.curriculum_stage || 1}`;
}

function updateSituation(state) {
    const nodes = state.graph?.nodes || [];
    const pending = nodes.filter(n => n.status === 'pending');
    const accepted = nodes.filter(n => n.status === 'accepted');
    const completed = nodes.filter(n => n.status === 'completed');
    const failed = nodes.filter(n => n.status === 'failed');

    if (pending.length > 0) {
        const topNode = pending[0];
        situationIcon.textContent = '⚡';
        situationText.innerHTML = `<strong>${pending.length} commitment${pending.length > 1 ? 's' : ''}</strong> need your decision. ` +
            `Next: <strong>"${topNode.label}"</strong> from ${topNode.stakeholder_id || 'unknown'}. ` +
            `${accepted.length} in progress, ${completed.length} completed.`;
    } else if (accepted.length > 0) {
        situationIcon.textContent = '⏳';
        situationText.innerHTML = `All reviewed! <strong>${accepted.length}</strong> task${accepted.length > 1 ? 's' : ''} in progress. ` +
            `Click <strong>"Wait"</strong> to advance time and let work complete.`;
    } else if (completed.length > 0) {
        situationIcon.textContent = '✅';
        situationText.innerHTML = `All done! <strong>${completed.length}</strong> completed, <strong>${failed.length}</strong> failed.`;
    } else {
        situationIcon.textContent = '💡';
        situationText.innerHTML = 'Click <strong>"New Episode"</strong> to start a new scenario.';
    }
}

function updateTargetInfo(state) {
    if (!selectedNode) {
        targetInfo.innerHTML = '<p class="muted">No commitment selected. Click a graph node or choose from the dropdown.</p>';
        return;
    }

    const node = state.graph?.nodes?.find(n => n.id === selectedNode);
    if (!node) {
        targetInfo.innerHTML = '<p class="muted">Node not found.</p>';
        return;
    }

    const statusColors = {
        pending: 'var(--accent-yellow)', accepted: 'var(--accent-blue)',
        completed: 'var(--accent-green)', failed: 'var(--accent-red)',
        at_risk: 'var(--accent-orange)',
    };
    const dlStr = node.deadline ? new Date(node.deadline).toLocaleString('en-US', {
        month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
    }) : 'None';

    targetInfo.innerHTML = `
        <div class="target-name">${node.label}</div>
        <div class="target-meta">
            <strong>ID:</strong> ${node.id} &nbsp; 
            <strong>Status:</strong> <span style="color:${statusColors[node.status] || '#888'}">${node.status}</span><br>
            <strong>Duration:</strong> ${node.estimated_duration_hours}h &nbsp;
            <strong>Deadline:</strong> ${dlStr}<br>
            <strong>From:</strong> ${node.stakeholder_id || '—'} &nbsp;
            <strong>Urgency:</strong> ${(node.urgency * 100).toFixed(0)}%
        </div>
    `;
}

function selectNode(nodeId) {
    selectedNode = nodeId;
    nodePicker.value = nodeId;

    // Update graph highlighting
    d3.selectAll('.node-circle').classed('selected', d => d.id === nodeId);

    if (currentState) updateTargetInfo(currentState);
}

// ═══════════════════════════════════════════════════════════
//  GRAPH RENDERING (D3.js)
// ═══════════════════════════════════════════════════════════

function updateGraph(state) {
    const container = document.getElementById('graph-area');
    const svg = d3.select('#graph-svg');
    svg.selectAll('*').remove();

    const nodes = state.graph?.nodes || [];
    const edges = state.graph?.edges || [];

    if (nodes.length === 0) {
        graphEmpty.style.display = 'flex';
        return;
    }
    graphEmpty.style.display = 'none';

    const W = container.clientWidth;
    const H = container.clientHeight;

    const g = svg.append('g');

    // Zoom
    const zoom = d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', (e) => g.attr('transform', e.transform));
    svg.call(zoom);

    // Status → icon mapping
    const statusIcon = {
        pending: '❓', accepted: '🔵', completed: '✅',
        failed: '❌', at_risk: '⚠️', renegotiated: '🔄',
    };

    // Build links
    const linkData = edges.map(e => ({
        source: e.source,
        target: e.target,
        type: e.type,
    }));

    // FORCE simulation
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(linkData).id(d => d.id).distance(160))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(W / 2, H / 2))
        .force('collision', d3.forceCollide(50));

    // Edges
    const link = g.selectAll('.edge-line')
        .data(linkData)
        .join('line')
        .attr('class', d => `edge-line ${d.type === 'dependency' ? 'dependency' : ''}`);

    // Node groups
    const nodeG = g.selectAll('.node-g')
        .data(nodes, d => d.id)
        .join('g')
        .attr('class', 'node-g')
        .call(d3.drag()
            .on('start', dragStart)
            .on('drag', dragged)
            .on('end', dragEnd));

    // Circle
    nodeG.append('circle')
        .attr('class', d => `node-circle ${d.status}${d.id === selectedNode ? ' selected' : ''}`)
        .attr('r', 22)
        .on('click', (e, d) => { e.stopPropagation(); selectNode(d.id); });

    // Icon inside circle
    nodeG.append('text')
        .attr('class', 'node-icon')
        .text(d => statusIcon[d.status] || '?');

    // Label below
    nodeG.append('text')
        .attr('class', 'node-label')
        .attr('dy', 38)
        .text(d => truncate(d.label, 22));

    // Meta below label
    nodeG.append('text')
        .attr('class', 'node-sub')
        .attr('dy', 52)
        .text(d => `${d.estimated_duration_hours}h · ${d.stakeholder_id || ''}`);

    // Tick
    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        nodeG.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Click background to deselect
    svg.on('click', () => {
        selectedNode = null;
        d3.selectAll('.node-circle').classed('selected', false);
        if (currentState) updateTargetInfo(currentState);
    });
}

function dragStart(e, d) {
    if (!e.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
}
function dragged(e, d) { d.fx = e.x; d.fy = e.y; }
function dragEnd(e, d) {
    if (!e.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
}

function truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.slice(0, max - 1) + '…' : str;
}

// ═══════════════════════════════════════════════════════════
//  NODE PICKER DROPDOWN
// ═══════════════════════════════════════════════════════════

function updateNodePicker(state) {
    const nodes = state.graph?.nodes || [];
    nodePicker.innerHTML = '<option value="">— Choose a commitment —</option>';
    nodes.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n.id;
        const statusIcon = { pending: '🟡', accepted: '🔵', completed: '🟢', failed: '🔴', at_risk: '🟠' };
        opt.textContent = `${statusIcon[n.status] || '⚪'} ${n.id} — ${n.label} [${n.status}]`;
        nodePicker.appendChild(opt);
    });
    if (selectedNode) nodePicker.value = selectedNode;
}

// ═══════════════════════════════════════════════════════════
//  TRUST
// ═══════════════════════════════════════════════════════════

function updateTrust(state) {
    // API returns trust_scores as {sid: float}, not trust_entries
    const scores = state.trust_scores || {};
    const multidim = state.multidim_trust || {};
    trustList.innerHTML = '';

    // If no trust_scores, try multidim_trust composite
    const trustData = Object.keys(scores).length > 0 ? scores : {};
    if (Object.keys(trustData).length === 0 && Object.keys(multidim).length > 0) {
        for (const [sid, md] of Object.entries(multidim)) {
            trustData[sid] = md.composite;
        }
    }

    for (const [sid, score] of Object.entries(trustData)) {
        const scoreClass = score > 0.6 ? 'high' : score > 0.35 ? 'mid' : 'low';
        const barColor = score > 0.6 ? 'var(--accent-green)' : score > 0.35 ? 'var(--accent-yellow)' : 'var(--accent-red)';
        const md = multidim[sid];

        let dimsHtml = '';
        if (md) {
            dimsHtml = `<div class="trust-dims">
                <span class="trust-dim-item"><span class="dim-dot r"></span> R ${md.reliability.toFixed(2)}</span>
                <span class="trust-dim-item"><span class="dim-dot c"></span> C ${md.competence.toFixed(2)}</span>
                <span class="trust-dim-item"><span class="dim-dot b"></span> B ${md.benevolence.toFixed(2)}</span>
            </div>`;
        }

        trustList.innerHTML += `
            <div class="trust-entry">
                <div class="trust-header">
                    <span class="trust-name">${sid}</span>
                    <span class="trust-score ${scoreClass}">${score.toFixed(2)}</span>
                </div>
                <div class="trust-bar-bg">
                    <div class="trust-bar-fill" style="width:${Math.max(0, score) * 100}%; background:${barColor}"></div>
                </div>
                ${dimsHtml}
            </div>
        `;
    }

    if (Object.keys(trustData).length === 0) {
        trustList.innerHTML = '<p class="muted" style="padding:4px 0">No trust data yet</p>';
    }
}

// ═══════════════════════════════════════════════════════════
//  REWARD
// ═══════════════════════════════════════════════════════════

function updateReward(stepData) {
    if (!stepData || stepData.reward === undefined) {
        rewardNumber.textContent = '—';
        rewardNumber.className = 'reward-big zero';
        rewardBreakdown.innerHTML = '<p class="muted" style="text-align:center;font-size:12px">Take an action to see reward</p>';
        return;
    }

    const r = stepData.reward;
    rewardNumber.textContent = (r >= 0 ? '+' : '') + r.toFixed(3);
    rewardNumber.className = `reward-big ${r > 0 ? 'positive' : r < 0 ? 'negative' : 'zero'}`;

    // Breakdown from info — try both key formats
    const components = stepData.info?.reward_components || {};
    const names = [
        [['fulfillment'], 'Fulfillment', 'var(--accent-green)'],
        [['trust_delta'], 'Trust Δ', 'var(--accent-blue)'],
        [['proactive'], 'Proactive', 'var(--accent-cyan)'],
        [['feasibility_acc', 'feasibility_accuracy'], 'Feasibility', 'var(--accent-purple)'],
        [['broken_penalty'], 'Broken', 'var(--accent-red)'],
        [['overrefusal_penalty', 'overrefusal'], 'Over-Refusal', 'var(--accent-orange)'],
        [['silent_drop_penalty', 'silent_drop'], 'Silent Drop', 'var(--accent-yellow)'],
    ];

    let html = '';
    for (const [keys, label, color] of names) {
        const val = keys.reduce((v, k) => v !== 0 ? v : (components[k] ?? 0), 0);
        const absVal = Math.abs(val);
        const width = Math.min(absVal * 200, 100);  // Scale for visibility
        const sign = val >= 0 ? '+' : '';
        const valColor = val > 0 ? 'var(--accent-green)' : val < 0 ? 'var(--accent-red)' : 'var(--text-dim)';

        html += `<div class="rw-row">
            <span class="rw-name">${label}</span>
            <div class="rw-bar"><div class="rw-fill" style="width:${width}%;background:${val >= 0 ? color : 'var(--accent-red)'}"></div></div>
            <span class="rw-val" style="color:${valColor}">${sign}${val.toFixed(3)}</span>
        </div>`;
    }
    rewardBreakdown.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════
//  EVENT LOG
// ═══════════════════════════════════════════════════════════

function logClear() { logList.innerHTML = ''; }

function logAdd(type, message) {
    const step = currentState?.step_number ?? 0;
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="log-step">${step}</span><span class="log-msg">${message}</span>`;
    logList.prepend(entry);

    // Keep max 50
    while (logList.children.length > 50) logList.lastChild.remove();
}

// ═══════════════════════════════════════════════════════════
//  AUTO-PLAY
// ═══════════════════════════════════════════════════════════

function toggleAutoplay() {
    if (autoplayTimer) {
        clearAutoplay();
    } else {
        if (!currentState) {
            logAdd('danger', 'Start an episode first!');
            return;
        }
        btnAuto.textContent = '⏸ Stop Auto-Play';
        btnAuto.classList.add('active');
        autoplayStep();
    }
}

function autoplayStep() {
    if (!currentState) { clearAutoplay(); return; }

    const nodes = currentState.graph?.nodes || [];
    const pending = nodes.filter(n => n.status === 'pending');

    let action = 'do_nothing';
    if (pending.length > 0) {
        selectNode(pending[0].id);
        // Simple heuristic: accept if urgency > 0.5, else counter
        action = pending[0].urgency > 0.5 ? 'accept' : 'counter_propose';
    }

    takeAction(action).then(() => {
        // Check if episode is still going
        if (currentState) {
            autoplayTimer = setTimeout(autoplayStep, 1200);
        }
    });
}

function clearAutoplay() {
    if (autoplayTimer) clearTimeout(autoplayTimer);
    autoplayTimer = null;
    btnAuto.textContent = '▶ Auto-Play Agent';
    btnAuto.classList.remove('active');
}

// ═══════════════════════════════════════════════════════════
//  EPISODE DONE
// ═══════════════════════════════════════════════════════════

function showEpisodeDone(stepData) {
    const nodes = currentState.graph?.nodes || [];
    const completed = nodes.filter(n => n.status === 'completed').length;
    const failed = nodes.filter(n => n.status === 'failed').length;
    const total = nodes.length;
    const fulfillment = total > 0 ? ((completed / total) * 100).toFixed(0) : 0;

    const overlay = document.createElement('div');
    overlay.id = 'episode-done';
    overlay.innerHTML = `
        <div class="done-card">
            <div class="done-title">${completed === total ? '🎉 Perfect Run!' : '🏁 Episode Complete'}</div>
            <div class="done-stats">
                <div class="done-stat">
                    <span class="done-stat-label">Total Reward</span>
                    <span class="done-stat-val" style="color:${totalReward >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${totalReward >= 0 ? '+' : ''}${totalReward.toFixed(3)}</span>
                </div>
                <div class="done-stat">
                    <span class="done-stat-label">Fulfillment</span>
                    <span class="done-stat-val">${fulfillment}% (${completed}/${total})</span>
                </div>
                <div class="done-stat">
                    <span class="done-stat-label">Failed</span>
                    <span class="done-stat-val" style="color:${failed > 0 ? 'var(--accent-red)' : 'var(--accent-green)'}">${failed}</span>
                </div>
            </div>
            <button class="start-btn" onclick="this.closest('#episode-done').remove()">Close</button>
        </div>
    `;
    document.body.appendChild(overlay);
}

// ═══════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════

function setActionsEnabled(enabled) {
    document.querySelectorAll('.act-btn').forEach(btn => btn.disabled = !enabled);
}

function shake(el) {
    el.style.animation = 'none';
    el.offsetHeight; // Trigger reflow
    el.style.animation = 'shake 0.4s ease';
}
