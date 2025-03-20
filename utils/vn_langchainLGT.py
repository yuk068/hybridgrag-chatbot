# Vietnamese translation for prompts

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents.base import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.runnables import RunnableConfig

examples = [
    {
        "text": (
            "Nguyễn Văn A là một kĩ sư phần mềm làm việc cho Microsoft từ 2009, "
            "anh ấy được coi là một nhân viên xuất sắc"
        ),
        "head": "Nguyễn Văn A",
        "head_type": "Cá nhân",
        "relation": "Làm việc cho",
        "tail": "Microsoft",
        "tail_type": "Công ti",
    },
    {
        "text": (
            "Nguyễn Văn A là một kĩ sư phần mềm làm việc cho Microsoft từ 2009, "
            "anh ấy được coi là một nhân viên xuất sắc"
        ),
        "head": "Nguyễn Văn A",
        "head_type": "Cá nhân",
        "relation": "Là",
        "tail": "Nhân viên xuất sắc",
        "tail_type": "Thành tựu",
    },
    {
        "text": (
            "Microsoft là một một công ti cung cấp"
            "nhiều sản phẩm thông dụng như Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Sản phẩm",
        "relation": "Sản xuất bởi",
        "tail": "Microsoft",
        "tail_type": "Công ti",
    },
    {
        "text": "Microsoft Word là một sản phẩm tối ưu có thể truy cập khi không trực tuyến",
        "head": "Microsoft Word",
        "head_type": "Sản phẩm",
        "relation": "Có đặc tính",
        "tail": "Sản phẩm tối ưu",
        "tail_type": "Đặc tính",
    },
    {
        "text": "Microsoft Word là một sản phẩm tối ưu có thể truy cập khi không trực tuyến",
        "head": "Microsoft Word",
        "head_type": "Sản phẩm",
        "relation": "Có đặc tính",
        "tail": "Có thể truy cập không trực tuyến",
        "tail_type": "Đặc tính",
    },
]

system_prompt = (
    "# Hướng dẫn tạo dựng hệ thống kiến thức\n"
    "## 1. Tổng quan\n"
    "Bạn là một thuật toán cao cấp với mục đích là trích xuất thông tin "
    "với định dạng có cấu trúc để xây dựng một đồ thị kiến thức.\n"
    "Bạn cần trích xuất tất cả thông tin từ văn bản một cách ngắn gọn "
    "mà không làm mất đi độ chính xác của thông tin. Tuyệt đối cấm thêm "
    "thông tin nào không được nhắc đến trực tiếp trong văn bản.\n"
    "- **Nodes** đại diện cho các đối tượng và khái niệm.\n"
    "- Mục đích là đạt được sự tối giản mà rõ ràng trong đồ thị kiến thức, đảm bảo dễ hiểu.\n"
    "## 2. Gán nhãn cho Nodes\n"
    "- **Đảm bảo chất lượng**: Hãy đảm bảo rằng bạn dùng các nhãn có sẵn cho các nodes.\n"
    "Đảm bảo bạn sử dụng các đối tượng và khái niệm trìu tượng cho các nhãn.\n"
    "- Ví dụ, nếu bạn xác định được một đối tượng đại diện cho một cá nhân, "
    "luôn gán nhãn node đó là **'Cá nhân'**. Tránh sử dụng các thuật ngữ cụ thể hơn như "
    "'Công nhân' hay 'Nhà báo'."
    "- **Node IDs**: Tuyệt đối cấm sử dụng số nguyên làm node IDs. Node IDs nên là các tên riêng "
    "hoặc các đối tượng, khái niệm có thể xác định được trong văn bản.\n"
    "- **Relationships** đại diện cho mối quan hệ giữa các đối tượng và khái niệm.\n"
    "Đảm bảo chất lượng và tính trìu tượng cho các relationships khi xây dựng đồ thị khiến thức.\n "
    "## 3. Đảm bảo bao quát\n"
    "- **Đảm bảo chất lượng cho các đối tượng và khái niệm**: Cần đảm bảo bao quát và chính xác.\n"
    'Nếu như một đối tượng, như "Phạm Thị B" được nhắc đến nhiều lần trong văn bản '
    'nhưng sử dụng các tên khác nhau hay xưng hô khác nhau (ví dụ, "Thị B", "anh ấy"), '
    "cần luôn sử dụng các loại định danh hoàn thiện nhất cho đối tượng đó trong đồ thị kiến thức. "
    'Trong trường hợp này, sử dụng "Phạm Thị B" là ID cho đối tượng.\n'
    "Đảm bảo rằng đồ thị kiến thức cần được rõ ràng và dễ hiểu, "
    "đảm bảo chất lượng và chính xác là tuyệt đối quan trọng.\n"
    "## 4. Tuân thủ hướng dẫn\n"
    "Tuân thủ tuyệt đối theo các hướng dẫn chặt chẽ. Việc sai lệch, "
    "không tuân thủ nghiêm ngặt sẽ dẫn đến hoạt động bị chấm dứt và kết thúc ngay lập tức."
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Lưu ý: Đưa câu trả lời theo đúng định dạng và cấm "
                "giải thích gì thêm. "
                "Sử dụng định dạng được cung cấp để trích xuất thông tin từ "
                "văn bản sau: {input}"
            ),
        ),
    ]
)


def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            "Đảm bảo bạn sử dụng các đối tượng và khái niệm trìu tượng cho các nhãn.\n"
            "- Ví dụ, nếu bạn xác định được một đối tượng đại diện cho một cá nhân, "
            "luôn gán nhãn node đó là **'Cá nhân'**. Tránh sử dụng các thuật ngữ cụ thể hơn như "
            "'Công nhân' hay 'Nhà báo'."
        )
    elif input_type == "relationship":
        return (
            "Đảm bảo chất lượng và tính trìu tượng cho các relationships khi xây dựng đồ thị khiến thức."
        )
    elif input_type == "property":
        return ""
    return ""


def optional_enum_field(
    enum_values: Optional[List[str]] = None,
    description: str = "",
    input_type: str = "node",
    llm_type: Optional[str] = None,
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    # Only openai supports enum param
    if enum_values and llm_type == "openai-chat":
        return Field(
            ...,
            enum=enum_values,
            description=f"{description}. Các lựa chọn bao gồm {enum_values}",
            **field_kwargs,
        )
    elif enum_values:
        return Field(
            ...,
            description=f"{description}. Các lựa chọn bao gồm {enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)



class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "trích xuất các head entity như Microsoft, Apple, John. "
            "Cần sử dụng các định danh mà con người có thể nhận dạng được."
        )
    )
    head_type: str = Field(
        description="nhãn của head entity, như Cá nhân, Công ti,..."
    )
    relation: str = Field(description="mối quan hệ giữa head entity và tail entity")
    tail: str = Field(
        description=(
            "trích xuất các tail entity như Microsoft, Apple, John. "
            "Cần sử dụng các định danh mà con người có thể nhận dạng được."
        )
    )
    tail_type: str = Field(
        description="nhãn của tail entity, như Cá nhân, Công ti,..."
    )



def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    base_string_parts = [
        "Bạn là một thuật toán cao cấp được thiết kế để trích xuất thông tin thành "
        "các định dạng có cấu trúc để xây dựng một đồ thị kiến thức. Công việc là "
        "xác định các entities và relations được yêu cầu bởi người dùng từ một văn bản. "
        "Bắt buộc phải cho câu trả lời theo định dạng JSON có chứa một danh sách với "
        'các đối tượng JSON. Mỗi đối tượng cần phải có các keys: "head", "head_type", '
        '"relation", "tail", và "tail_type". "head" cần phải chứa văn bản của đối tượng'
        "được trích xuất với một trong các loại từ danh sách được cung cấp từ yêu cầu của "
        'người dùng. "head_type" cần phải chứa loại của đối tượng head entity được trích xuất, '
        f'"head_type" cần phải là một trong các loại sau: {node_labels_str}.'
        if node_labels
        else "",
        '"relation" cần phải chứa loại relation giữa "head" và "tail" '
        f'"relation" cần phải là một trong các loại sau: {rel_types_str}.'
        if rel_types
        else "",
        '"tail" cần phải chứa văn bản của đối tượng được trích xuất là tail của relation, '
        f'"tail_type" cần phải chứa các loại của tail entity từ danh sách: {node_labels_str}.'
        if node_labels
        else "",
        "Trích xuất nhiều entities và relations nhất có thể. Cần đảm bảo "
        "chất lượng: khi trích xuất entities, cần đảm bảo chất lượng và độ chính xác. "
        'Nếu như một đối tượng, như "Phạm Thị B" được nhắc đến nhiều lần trong văn bản '
        'nhưng sử dụng các tên khác nhau hay xưng hô khác nhau (ví dụ, "Thị B", "anh ấy"), '
        "cần luôn sử dụng các loại định danh hoàn thiện nhất cho đối tượng đó trong đồ thị kiến thức. "
        "Đồ thị kiến thức cần được rõ ràng, dễ hiểu, đảm bảo chất lượng và độ chính xác. "
        "Cần phải đảm bảo chất lượng của các entities và relations.\nTuyệt đối cấm giải thích gì thêm."
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))
    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        "Dựa vào ví dụ trên, hãy trích xuất các entities và relations "
        "từ văn bản được cung cấp.\n\n",
        "Sử dụng các entity types sau, cấm sử dụng các loại entity "
        "không có trong danh sách sau:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Sử dụng các relation types sau, cấm sử dụng các loại relation "
        "không có trong danh sách sau:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
        "Sau đây là một vài ví dụ cho các văn bản và các entities và relationships "
        "được trích xuất:"
        "{examples}\n"
        "Đối với văn bản sau, trích xuất các entities và relations theo cách giống với "
        "các ví dụ được cung cấp."
        "{format_instructions}\nVăn bản cần được trích xuất: {input}",
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt



def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    Create a simple graph model with optional constraints on node
    and relationship types.

    Args:
        node_labels (Optional[List[str]]): Specifies the allowed node types.
            Defaults to None, allowing all node types.
        rel_types (Optional[List[str]]): Specifies the allowed relationship types.
            Defaults to None, allowing all relationship types.
        node_properties (Union[bool, List[str]]): Specifies if node properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        relationship_properties (Union[bool, List[str]]): Specifies if relationship
            properties should be included. If a list is provided, only properties with
            keys in the list will be included. If True, all properties are included.
            Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[_Graph]: A graph model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the node or relationship properties list.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier."),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
    }

    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of node properties"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    relationship_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of source node",
            ),
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the source node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of target node",
            ),
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the target node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                rel_types,
                description="The type of the relationship.",
                input_type="relationship",
                llm_type=llm_type,
            ),
        ),
    }
    if relationship_properties:
        if (
            isinstance(relationship_properties, list)
            and "id" in relationship_properties
        ):
            raise ValueError(
                "The relationship property 'id' is reserved and cannot be used."
            )
        # Map True to empty array
        relationship_properties_mapped: List[str] = (
            [] if relationship_properties is True else relationship_properties
        )

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                relationship_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        relationship_fields["properties"] = (
            Optional[List[RelationshipProperty]],
            Field(None, description="List of relationship properties"),
        )
    SimpleRelationship = create_model("SimpleRelationship", **relationship_fields)  # type: ignore

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

    return DynamicGraph



def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)



def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    properties = {}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value
    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )



def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        node_properties = {}
        if "properties" in node and node["properties"]:
            for p in node["properties"]:
                node_properties[format_property_key(p["key"])] = p["value"]
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type", "Node"),
                properties=node_properties,
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = None
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = None

        rel_properties = {}
        if "properties" in rel and rel["properties"]:
            for p in rel["properties"]:
                rel_properties[format_property_key(p["key"])] = p["value"]

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
                properties=rel_properties,
            )
        )
    return nodes, relationships


def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize()  # type: ignore[arg-type]
            if el.type
            else None,  # handle empty strings  # type: ignore[arg-type]
            properties=el.properties,
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in rels
    ]


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)



def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                try:
                    argument_json = json.loads(
                        raw_schema["raw"].additional_kwargs["function_call"][
                            "arguments"
                        ]
                    )
                except Exception:  # Ollama type response
                    argument_json = raw_schema["raw"].tool_calls[0]["args"]
                    if isinstance(argument_json["nodes"], str):
                        argument_json["nodes"] = json.loads(argument_json["nodes"])
                    if isinstance(argument_json["relationships"], str):
                        argument_json["relationships"] = json.loads(
                            argument_json["relationships"]
                        )

            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]
        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes if node.id]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [
                map_to_base_relationship(rel)
                for rel in parsed_schema.relationships
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


class LLMGraphTransformer:
    """
    Transform documents into graph-based documents using a LLM.

    It allows specifying constraints on the types of nodes and relationships to include
    in the output graph. The class supports extracting properties for both nodes and
    relationships.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured
          output.
        allowed_nodes (List[str], optional): Specifies which node types are
          allowed in the graph. Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types
          are allowed in the graph. Defaults to an empty list, allowing all relationship
          types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to
          the LLM with additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply
          filtering to strictly adhere to `allowed_nodes` and `allowed_relationships`.
          Defaults to True.
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any
          node properties from text. Alternatively, a list of valid properties can
          be provided for the LLM to extract, restricting extraction to those specified.
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract
          any relationship properties from text. Alternatively, a list of valid
          properties can be provided for the LLM to extract, restricting extraction to
          those specified.
        ignore_tool_usage (bool): Indicates whether the transformer should
          bypass the use of structured output functionality of the language model.
          If set to True, the transformer will not use the language model's native
          function calling capabilities to handle structured output. Defaults to False.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm=ChatOpenAI(temperature=0)
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Person", "Organization"])

            doc = Document(page_content="Elon Musk is suing OpenAI")
            graph_documents = transformer.convert_to_graph_documents([doc])
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
    ) -> None:
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = not ignore_tool_usage
        # Check if the LLM really supports structured output
        if self._function_call:
            try:
                llm.with_structured_output(_Graph)
            except NotImplementedError:
                self._function_call = False
        if not self._function_call:
            if node_properties or relationship_properties:
                raise ValueError(
                    "The 'node_properties' and 'relationship_properties' parameters "
                    "cannot be used in combination with a LLM that doesn't support "
                    "native function calling."
                )
            try:
                import json_repair  # type: ignore

                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes, allowed_relationships
            )
            self.chain = prompt | llm
        else:
            # Define chain
            try:
                llm_type = llm._llm_type  # type: ignore
            except AttributeError:
                llm_type = None
            schema = create_simple_model(
                allowed_nodes,
                allowed_relationships,
                node_properties,
                llm_type,
                relationship_properties,
            )
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or default_prompt
            self.chain = prompt | structured_llm


    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        raw_schema = self.chain.invoke({"input": text}, config=config)
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            nodes_set = set()
            relationships = []
            
            if not isinstance(raw_schema, str):
                raw_schema = raw_schema.content
                
            parsed_json = self.json_repair.loads(raw_schema)
            if isinstance(parsed_json, dict):
                parsed_json = [parsed_json]
            for rel in parsed_json:
                if not isinstance(rel, dict):
                    continue
                
                # Extract values safely
                head = str(rel.get("head", "")).strip()
                tail = str(rel.get("tail", "")).strip()
                head_type = str(rel.get("head_type", "")).strip()
                tail_type = str(rel.get("tail_type", "")).strip()
                relation = str(rel.get("relation", "")).strip()
            
                # Skip this relationship if head or tail is missing
                if not head or not tail or not head_type or not tail_type:
                    continue  
            
                # Apply replacements for spaces and problematic characters
                head = re.sub(r"[\/,\'\"]", "_", head.replace(" ", "_"))
                tail = re.sub(r"[\/,\'\"]", "_", tail.replace(" ", "_"))
                head_type = re.sub(r"[\/,\'\"]", "_", head_type.replace(" ", "_"))
                tail_type = re.sub(r"[\/,\'\"]", "_", tail_type.replace(" ", "_"))
                relation = re.sub(r"[\/,\'\"]", "_", relation.replace(" ", "_"))
            
                # Nodes need to be deduplicated using a set
                nodes_set.add((head, head_type))
                nodes_set.add((tail, tail_type))
            
                source_node = Node(id=head, type=head_type)
                target_node = Node(id=tail, type=tail_type)
            
                # Skip if the relation is empty
                if relation:
                    relationships.append(
                        Relationship(
                            source=source_node, target=target_node, type=relation
                        )
                    )

            # Create nodes list
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]

        # Strict mode filtering
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)


    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document, config) for document in documents]


    async def aprocess_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Asynchronously processes a single document, transforming it into a
        graph document.
        """
        text = document.page_content
        raw_schema = await self.chain.ainvoke({"input": text}, config=config)
        raw_schema = cast(Dict[Any, Any], raw_schema)
        nodes, relationships = _convert_to_graph_document(raw_schema)

        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)


    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = [
            asyncio.create_task(self.aprocess_response(document, config))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        return results