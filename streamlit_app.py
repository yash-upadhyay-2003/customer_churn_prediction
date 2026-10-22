                    err_json = resp.json()
                    st.session_state.last_error = f"API Error ({resp.status_code}): {err_json.get('detail', resp.text)}"
                except Exception:
                    st.session_state.last_error = f"API Error ({resp.status_code}): {resp.text}"
        except requests.exceptions.ConnectionError:
            st.session_state.prediction_result = None
            st.session_state.last_error = (
                f"Connection failure: unable to reach prediction service at {effective_api_url}. "
                "Ensure FastAPI backend is running via `python -m uvicorn app.main:app --reload`."
            )
        except requests.exceptions.Timeout:
            st.session_state.prediction_result = None
            st.session_state.last_error = "Request timed out while waiting for inference response."
        except Exception as e:
            st.session_state.prediction_result = None
            st.session_state.last_error = f"Unexpected client error: {str(e)}"

# =============================================================================
# RIGHT COLUMN: Prediction Workspace
# =============================================================================
with col_pred:
    st.markdown('<div class="section-eyebrow">PREDICTION WORKSPACE</div>', unsafe_allow_html=True)

    result = st.session_state.prediction_result
    error_msg = st.session_state.last_error

    if error_msg:
        st.markdown(
            textwrap.dedent(f"""
                <div class="workspace-box">
                    <div class="result-badge badge-churn">SERVICE NOTICE</div>
                    <div class="workspace-empty-title">Inference Unavailable</div>
                    <div class="workspace-empty-desc">{error_msg}</div>
                    <div class="empty-graphic-guide">
                        Expected API endpoint:<br><code>{effective_api_url}/predict</code>
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    elif result is None:
        # Empty State
        st.markdown(
            textwrap.dedent("""
                <div class="workspace-box">
                    <div class="result-badge" style="background: #FFFFFF; border: 1px solid #EAE6E3; color: #77716F;">
                        STANDBY
                    </div>
                    <div class="workspace-empty-title">Awaiting customer profile</div>
                    <div class="workspace-empty-desc">
                        Complete the profile and run the model to view the estimated churn outcome.
                    </div>
                    <div class="empty-graphic-guide">
                        15 features configured<br>
                        Machine Learning Pipeline Ready
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    else:
        # Active Prediction Outcome
        pred_label = result.get("prediction", "Unknown")
        churn_prob = float(result.get("churn_probability", 0.0))
        prob_pct = int(round(churn_prob * 100))
        is_churn = (pred_label == "Churn") or (result.get("predicted_class") == 1)

        badge_class = "badge-churn" if is_churn else "badge-no-churn"
        badge_text = "ELEVATED CHURN RISK" if is_churn else "HEALTHY RETENTION"
        bar_fill_class = "prob-bar-fill-churn" if is_churn else "prob-bar-fill-safe"

        if is_churn:
            interpretation_text = (
                f"The model estimated a <strong>{prob_pct}% probability of churn</strong>. "
                "Account attributes (such as usage fluctuation, penalties, or service inquiries) place this "
                "customer in an at-risk cluster."
            )
        else:
            interpretation_text = (
                f"The model estimated a <strong>{prob_pct}% probability of churn</strong>. "
                "Current tenure, spend, and satisfaction patterns indicate steady account continuity."
            )

        result_html = textwrap.dedent(f"""
            <div class="workspace-box">
                <div class="result-badge {badge_class}">{badge_text}</div>
                <div class="result-outcome">{pred_label}</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; color: #77716F; text-transform: uppercase;">
                        Model Estimate
                    </span>
                    <span class="probability-number">{prob_pct}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="{bar_fill_class}" style="width: {max(prob_pct, 2)}%;"></div>
                </div>
                <div class="result-interpretation">
                
                    {interpretation_text}
                </div>
                <div class="result-caption">
                    Model estimate based on trained Machine Learning pipeline; not a calibrated risk score or guarantee of future behaviour.
                </div>
            </div>
        """).strip()
        st.markdown(result_html, unsafe_allow_html=True)

        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("Run another analysis"):
            st.session_state.prediction_result = None
            st.session_state.last_error = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Compact Editorial Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <div class="footer-left">
            Customer Churn Prediction · Machine Learning
        </div>
        <div class="footer-right">
            SCIKIT-LEARN PIPELINE · FASTAPI BACKEND
        </div>
    </div>
    """,
