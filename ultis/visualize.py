
import numpy as np # Import NumPy / Nhập thư viện NumPy
try:
    import networkx as nx # Import NetworkX for graph logic / Nhập NetworkX cho logic đồ thị
    import matplotlib.pyplot as plt # Import Matplotlib for plotting / Nhập Matplotlib để vẽ hình
    import matplotlib.colors as mcolors # Import color utilities / Nhập các tiện ích màu sắc
except ImportError:
    nx = None # Handle missing packages / Xử lý khi thiếu thư viện
    plt = None

def plot_dag(W_matrix, labels=None, GT_matrix=None, ate_matrix=None, metrics=None, title="CausalFlowNet Discovery Graph",
             threshold=0.1, ax=None, save_path=None, node_size=2000,
             font_size=10, figure_size=(12, 9)):
    """
    Render a weighted causal DAG as a directed graph. / Vẽ đồ thị DAG nhân quả có trọng số.
    Supports comparison with Ground Truth if GT_matrix is provided. / Hỗ trợ so sánh với nhãn thực tế (Ground Truth).
    Also supports displaying ATE labels on edges if ate_matrix is provided. / Hỗ trợ hiển thị nhãn ATE nếu có ate_matrix.
    Display performance metrics (TPR, FPR, etc.) if metrics dict is provided. / Hiển thị các chỉ số hiệu năng nếu có metrics.
    """
    if nx is None or plt is None:
        raise ImportError("Install required packages: pip install networkx matplotlib")

    # Filter edges below threshold / Lọc bỏ các cạnh có trọng số tuyệt đối dưới ngưỡng (threshold)
    W_filtered = np.where(np.abs(W_matrix) > threshold, W_matrix, 0)
    G = nx.DiGraph(W_filtered) # Create directed graph from weights / Tạo đồ thị có hướng từ trọng số

    n_nodes = W_matrix.shape[0] # Number of nodes / Số lượng nút
    if labels is None:
        labels = [f"X{i}" for i in range(n_nodes)] # Default labels / Nhãn mặc định X0, X1...
    
    # Map node indices to labels / Ánh xạ chỉ số nút sang nhãn văn bản
    G = nx.relabel_nodes(G, {i: labels[i] for i in range(n_nodes)})

    # Initialize Ground Truth graph if provided / Khởi tạo đồ thị thực tế nếu có
    G_gt = None
    if GT_matrix is not None:
        G_gt = nx.DiGraph(np.abs(GT_matrix) > 0.1)
        G_gt = nx.relabel_nodes(G_gt, {i: labels[i] for i in range(n_nodes)})

    show_plot = False
    if ax is None:
        # Create new figure if no axes provided / Tạo khung hình mới nếu chưa có
        fig, ax = plt.subplots(figsize=figure_size)
        show_plot = True

    if len(G.edges) == 0:
        # Handle case with no discovered edges / Xử lý trường hợp không tìm thấy cạnh nào
        ax.text(0.5, 0.5, "No Edges Found Above Threshold", horizontalalignment='center',
                verticalalignment='center', fontsize=20, color='red', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        if show_plot:
            if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
            else: plt.show()
        return

    # Categorize edges and determine styles / Phân loại cạnh và xác định phong cách vẽ
    edges = list(G.edges()) # Get all discovered edges / Lấy danh sách các cạnh tìm được
    edge_colors = [] # Storage for colors / Lưu trữ màu sắc
    edge_styles = [] # Storage for dash styles / Lưu trữ kiểu nét vẽ
    
    # Premium Color Palette / Bảng màu cao cấp
    COLOR_CORRECT = "#c0392b"    # Red (Correct) / Đỏ (Đúng)
    COLOR_INDIRECT = "#2980b9"   # Blue (Indirect) / Xanh dương (Gián tiếp)
    COLOR_REVERSED = "#f39c12"   # Orange (Reversed) / Cam (Ngược hướng)
    COLOR_UNEXPLAINED = "#27ae60" # Green (False Positive) / Xanh lá (Sai/Dư)

    if G_gt is not None:
        # Compare with Ground Truth / So sánh với nhãn thực tế
        for u, v in edges:
            if G_gt.has_edge(u, v): # Correct edge / Cạnh đúng
                edge_colors.append(COLOR_CORRECT)
                edge_styles.append("solid")
            elif G_gt.has_edge(v, u): # Reversed direction / Cạnh bị ngược
                edge_colors.append(COLOR_REVERSED)
                edge_styles.append("solid")
            elif nx.has_path(G_gt, u, v): # Indirect path (Grandparent, etc.) / Đường đi gián tiếp
                edge_colors.append(COLOR_INDIRECT)
                edge_styles.append("dashed")
            else: # Completely wrong / Sai hoàn toàn
                edge_colors.append(COLOR_UNEXPLAINED)
                edge_styles.append("dashed")
    else:
        # Default behavior: Color by effect strength / Mặc định: Tô màu theo cường độ tác động
        weights = [abs(G[u][v]['weight']) for u, v in edges]
        max_w = max(weights) if weights else 1.0
        norm = mcolors.Normalize(vmin=0, vmax=max_w) # Normalize weights / Chuẩn hóa trọng số
        cmap = plt.cm.winter_r # Winter colormap / Bảng màu mùa đông
        edge_colors = [cmap(norm(w)) for w in weights]
        edge_styles = ["solid"] * len(edges)

    # Calculate layout (Spring Layout for optimal spacing) / Tính toán bố cục (Spring Layout để giãn cách tối ưu)
    pos = nx.spring_layout(G, k=1.8, iterations=150, seed=42)

    # Draw nodes (Modern oval-like style) / Vẽ các nút (Phong cách hình trứng hiện đại)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color='#d6eaf8', # Light blue fill / Màu nền xanh nhạt
                           edgecolors='#2e86c1', # Dark blue border / Viền xanh đậm
                           linewidths=2.0)
    
    # Draw labels / Vẽ nhãn tên nút
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                            font_weight='bold', font_color='#1b2631')
    
    # Draw edges with specific patterns / Vẽ các cạnh với các kiểu dáng đặc thù
    for i, (u, v) in enumerate(edges):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, 
                               arrowstyle='-|>', arrowsize=20, # Directed arrows / Mũi tên chỉ hướng
                               edge_color=edge_colors[i],
                               style=edge_styles[i],
                               width=2.5,
                               connectionstyle="arc3,rad=0.1", # Slight curve for better visibility / Bo cong nhẹ để dễ nhìn
                               node_size=node_size, alpha=0.9)
    
    # NEW: Draw ATE labels on edges / MỚI: Vẽ nhãn ATE trên các cạnh
    if ate_matrix is not None:
        edge_labels = {}
        # Need to map back labels to indices to access ate_matrix
        label_to_idx = {label: i for i, label in enumerate(labels)}
        for u, v in edges:
            u_idx = label_to_idx[u]
            v_idx = label_to_idx[v]
            ate_val = ate_matrix[u_idx, v_idx]
            if abs(ate_val) > 0.001:
                edge_labels[(u, v)] = f"{ate_val:+.2f}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                     font_size=font_size-2, font_color="#154360",
                                     font_weight='bold', label_pos=0.6,
                                     rotate=True, clip_on=False)

    # NEW: Display metrics on the plot / MỚI: Hiển thị chỉ số hiệu năng trên hình
    if metrics:
        metrics_text = (
            f"TPR: {metrics.get('tpr', 0):.2f}\n"
            f"FPR: {metrics.get('fpr', 0):.2f}\n"
            f"SHD: {metrics.get('shd', 0)}\n"
            f"SID: {metrics.get('sid', 0)}"
        )
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                fontsize=font_size + 2, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#2e86c1'))

    # Formatting / Định dạng hình ảnh
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    ax.axis('off') # Hide axes / Ẩn các trục tọa độ

    # Add Legend for edge categories / Thêm bảng chú giải cho các loại cạnh
    if G_gt is not None:
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=COLOR_CORRECT, lw=2, linestyle='-'),
            Line2D([0], [0], color=COLOR_INDIRECT, lw=2, linestyle='--'),
            Line2D([0], [0], color=COLOR_REVERSED, lw=2, linestyle='-'),
            Line2D([0], [0], color=COLOR_UNEXPLAINED, lw=2, linestyle='--')
        ]
        ax.legend(custom_lines, ['Correct Edge', 'Indirect Edge', 'Reversed Edge', 'Unexplained Edge'],
                  loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=11, title="Edge Categories")
    else:
        # Add colorbar for weights / Thêm thanh chỉ số màu cho trọng số
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Causal Effect Strength (Weight)', rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()

    # Save logic / Logic lưu tệp tin
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
        print(f"Causal graph comparison saved to: {save_path}")
        
    if show_plot:
        if save_path is None: plt.show()
        else: plt.close()

def plot_structure_comparison(W_matrix, GT_matrix, labels=None, title="Causal Discovery Comparison",
                             threshold=0.1, save_path=None, figure_size=(20, 10)):
    """
    Generate a side-by-side comparison of the Ground Truth graph and the Discovery graph.
    Tạo hình ảnh so sánh song song giữa đồ thị Ground Truth và đồ thị khám phá được.
    """
    if nx is None or plt is None:
        raise ImportError("Install required packages: pip install networkx matplotlib")

    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    
    n_nodes = W_matrix.shape[0]
    if labels is None:
        labels = [f"X{i}" for i in range(n_nodes)]

    # Create graphs to compute a shared layout / Tạo đồ thị để tính toán bố cục chung
    G_est = nx.DiGraph(np.where(np.abs(W_matrix) > threshold, W_matrix, 0))
    G_gt = nx.DiGraph(np.abs(GT_matrix) > 0.1)
    
    # Combined graph for layout stability / Kết hợp đồ thị để ổn định bố cục
    G_combined = nx.compose(G_est, G_gt)
    pos = nx.spring_layout(G_combined, k=2.0, iterations=150, seed=42)
    
    # Create mapping dictionary for drawing / Tạo từ điển ánh xạ để vẽ
    node_mapping = {i: labels[i] for i in range(n_nodes)}

    # 1. Plot Ground Truth (Left) / Vẽ Ground Truth (Bên trái)
    ax_gt = axes[0]
    G_gt_labeled = nx.relabel_nodes(G_gt, node_mapping)
    pos_labeled = {labels[i]: pos[i] for i in range(n_nodes)}
    
    # Elegant high-contrast styling: White nodes, Black edges / Phong cách tương phản cao: Nút trắng, Cạnh đen
    nx.draw_networkx_nodes(G_gt_labeled, pos_labeled, ax=ax_gt, node_size=1800, 
                           node_color='#ffffff', edgecolors='#000000', linewidths=3.0)
    nx.draw_networkx_labels(G_gt_labeled, pos_labeled, ax=ax_gt, font_size=11, font_weight='bold')
    
    # Draw solid black arrows for absolute clarity / Vẽ mũi tên đen đặc để đảm bảo sự rõ ràng tuyệt đối
    nx.draw_networkx_edges(G_gt_labeled, pos_labeled, ax=ax_gt, arrowstyle='-|>', 
                           arrowsize=25, edge_color='#000000', width=3.0,
                           connectionstyle="arc3,rad=0.1", alpha=1.0)
    
    ax_gt.set_title("🟢 GROUND TRUTH (TARGET)\n(Cấu trúc thực tế chuẩn - Đối chứng)", fontsize=16, fontweight='bold', pad=20)
    ax_gt.axis('off')
    
    # Add a vertical divider / Thêm đường kẻ ngăn cách
    # Draw line in figure coordinates to ensure it's centered
    from matplotlib.lines import Line2D
    line = Line2D([0.5, 0.5], [0.1, 0.9], transform=fig.transFigure, color='black', lw=1.5, ls='--')
    fig.lines.append(line)

    # 2. Plot Discovery (Right) / Vẽ kết quả khám phá (Bên phải)
    ax_est = axes[1]
    plot_dag(W_matrix, labels=labels, GT_matrix=GT_matrix, 
             title="🔴 DISCOVERED STRUCTURE\n(Cấu trúc tìm được - Phân loại màu)", 
             threshold=threshold, ax=ax_est)
    
    plt.suptitle(title, fontsize=22, fontweight='bold', y=1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Comparison plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()
